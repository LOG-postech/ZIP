import os.path as osp
from this import d
import time
import datetime
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
import os

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, set_random_seed, AverageMeter, MetricMeter
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers import visual_prompters
from trainers.utils import clip_clipping, load_clip_to_cpu
import numpy as np
from math import sqrt
from trainers.zsclip import CUSTOM_TEMPLATES
from torchinfo import summary
import open_clip
_tokenizer = _Tokenizer()


class CustomCLIP(nn.Module):
    '''editted for visual prompting'''
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.token_embedding.weight.dtype
        self.p_eps = cfg.TRAINER.BLACKVIP.P_EPS

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        self.n_classes = len(classnames)
        self.classnames = classnames
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features.to(device)
        self.coordinator = visual_prompters.__dict__[cfg.TRAINER.BLACKVIP.METHOD](cfg, 'BLACKVIP')
 
    def forward(self, image):
        prompt, _  = self.coordinator(image.type(self.dtype))
        prompted_images = clip_clipping(image + self.p_eps * prompt)
        image_features = self.image_encoder(prompted_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class BlackVIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BLACKVIP.PREC == "fp32" or cfg.TRAINER.BLACKVIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model = clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        #! blackbox setting
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.coordinator.dec, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.coordinator.dec, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("coordinator", self.model.coordinator.dec, self.optim, self.sched)
        self.N_params = len(torch.nn.utils.parameters_to_vector(self.model.coordinator.dec.parameters()))

        # print model.intrinsic_learner summary
        for name, param in self.model.coordinator.dec.named_parameters():
            print(name, param.size())

        print(f"Number of optimization variables: {self.N_params}")

        #! BlackVIP parameters
        self.opt_type                                  = cfg.TRAINER.BASIC.OPT_TYPE
        self.sp_avg                                    = cfg.TRAINER.BASIC.SP_AVG
        self.o, self.c, self.a, self.alpha, self.gamma = cfg.TRAINER.BLACKVIP.SPSA_PARAMS
        self.b1                                        = cfg.TRAINER.BLACKVIP.MOMS

        self.step = 0
        self.m1 = 0
        self.loss_fn = F.cross_entropy
        self.API_calls = 0
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        with torch.no_grad():
            image, label = self.parse_batch_train(batch)
            with autocast(): #! for fast training
                #* SPSA scheduling
                ak = self.a/((self.step + self.o)**self.alpha)
                ck = self.c/(self.step**self.gamma)
                
                # gradient estimation
                w = torch.nn.utils.parameters_to_vector(self.model.coordinator.dec.parameters())
                ghat, loss, acc = self.spsa_grad_estimate_bi(self.sp_avg, w, image, label, ck)

                if self.opt_type == 'spsa-gc':
                    if self.step > 1:  self.m1 = self.b1*self.m1 + ghat
                    else:              self.m1 = ghat
                    accum_ghat = ghat + self.b1*self.m1
                elif self.opt_type == 'spsa':
                    accum_ghat = ghat
                else:
                    raise ValueError

                #* param update
                w_new = w - ak * accum_ghat
                torch.nn.utils.vector_to_parameters(w_new, self.model.coordinator.dec.parameters())

        loss_summary = {"loss": loss,"acc": acc,}
        return loss_summary

    def spsa_grad_estimate_bi(self, sp_avg, w, image, label, ck):
        #* repeat k times and average them for stabilizing
        ghats = []
        for spk in range(sp_avg):
            #! Bernoulli {-1, 1}
            # perturb = torch.bernoulli(torch.empty(self.N_params).uniform_(0,1)).cuda()
            # perturb[perturb < 1] = -1
            #! Segmented Uniform [-1, 0.5] U [0.5, 1]
            p_side = (torch.rand(self.N_params).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side,-p_side], dim=1)
            perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).cuda()
            del samples; del p_side

            #* two-side Approximated Numerical Gradient
            w_r = w + ck*perturb
            w_l = w - ck*perturb
            
            torch.nn.utils.vector_to_parameters(w_r, self.model.coordinator.dec.parameters())
            output1 = self.model(image)
            torch.nn.utils.vector_to_parameters(w_l, self.model.coordinator.dec.parameters())
            output2 = self.model(image)
            loss1 = self.loss_fn(output1, label)
            loss2 = self.loss_fn(output2, label)
            self.API_calls += 2

            #* parameter update via estimated gradient
            ghat = (loss1 - loss2)/((2*ck)*perturb)
            ghats.append(ghat.reshape(1, -1))
        if sp_avg == 1: pass
        else: ghat = torch.cat(ghats, dim=0).mean(dim=0) 
        loss = ((loss1+loss2)/2)
        acc = ((compute_accuracy(output1, label)[0]+
                compute_accuracy(output2, label)[0])/2).item()

        return ghat, loss, acc

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            self.step += 1
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()

            if self.cfg.TRAINER.BASIC.MAX_API_CALLS and self.API_calls >= self.cfg.TRAINER.BASIC.MAX_API_CALLS:
                break

        #* added part
        print(self.API_calls)
        epoch_acc = losses.meters['acc'].avg
        epoch_loss = losses.meters['loss'].avg

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")

            #* changed part
            result = self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        curr_result = 0.0
        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if self.cfg.TRAINER.BASIC.MAX_API_CALLS and self.API_calls >= self.cfg.TRAINER.BASIC.MAX_API_CALLS:
            print("Reach the maximum number of API calls")
            self.epoch = self.max_epoch - 1
            last_epoch = True

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if self.cfg.TRAINER.BASIC.MAX_API_CALLS and self.API_calls >= self.cfg.TRAINER.BASIC.MAX_API_CALLS:
                break
        self.after_train()