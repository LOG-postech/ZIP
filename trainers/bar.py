import os.path as osp
from termios import VLNEXT
import time
import datetime
from math import sqrt 
import os 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, set_random_seed, AverageMeter, MetricMeter, mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.utils import FocalLoss, load_clip_to_cpu
from trainers import visual_prompters
from tqdm import tqdm
import numpy as np
from trainers.zsclip import CUSTOM_TEMPLATES
_tokenizer = _Tokenizer()


class CustomCLIP(nn.Module):
    '''editted for visual prompting'''
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features.to(device)

        self.program = visual_prompters.__dict__[cfg.TRAINER.BAR.METHOD](cfg)

    def forward(self, image):
        programmed_image = self.program(image.type(self.dtype))
        image_features = self.image_encoder(programmed_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class BAR(TrainerX):
    """Black-Box Adversarial Reprogramming
    """
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model = clip_model.float()
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.program, cfg.MODEL.INIT_WEIGHTS)

        #! blackbox setting
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.program.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("program", self.model.program, self.optim, self.sched)
        self.N_params = len(torch.nn.utils.parameters_to_vector(self.model.program.parameters()))

        print(f"Number of optimization variables: {self.N_params}")

        #! BAR parameters
        self.sp_avg                = cfg.TRAINER.BASIC.SP_AVG
        self.init_lr, self.min_lr  = cfg.TRAINER.BAR.LRS
        self.beta                  = cfg.TRAINER.BAR.SMOOTH
        self.sigma                 = cfg.TRAINER.BAR.SIMGA        
        self.decaying_steps        = cfg.TRAINER.BAR.DECAYING
        
        self.step = 0                  
        self.step_for_pdecay = 0

        #! BAR's default loss
        self.loss_fn = FocalLoss(cfg.TRAINER.BAR.FOCAL_G)
        self.tot_itert = 0
        self.best_result = -1.0
        self.API_calls = 0

    def forward_backward(self, batch):
        with torch.no_grad():
            image, label = self.parse_batch_train(batch)
            #* learning rate scheduling
            decay_steps = self.total_length * self.decaying_steps
            self.step_for_pdecay = min(self.step_for_pdecay, decay_steps)
            ak = (self.init_lr - self.min_lr) * (1 - self.step_for_pdecay / decay_steps) ** (self.decaying_steps) + self.min_lr
            
            #* prompt parameters
            w = torch.nn.utils.parameters_to_vector(self.model.program.parameters())

            #* Randomized Gradient-Free Minimization
            m, sigma = 0, self.sigma 
            beta = torch.tensor(self.beta).cuda()
            q = torch.tensor(self.sp_avg).cuda()
            d = self.N_params

            output = self.model(image)
            self.API_calls += 1
            loss_pivot = self.loss_fn(output, label)

            ghat = torch.zeros(d).cuda()
            for _ in range(self.sp_avg):
                # Obtain a random direction vector
                u = torch.normal(m, sigma, size=(d,)).cuda()
                u = u / torch.norm(u, p=2)

                # Forward evaluation 
                w_r = w + beta * u
                torch.nn.utils.vector_to_parameters(w_r, self.model.program.parameters())

                # Gradient estimation
                output_pt = self.model(image)
                self.API_calls += 1
                loss_pt = self.loss_fn(output_pt, label)
                ghat = ghat + (d / q) * u * (loss_pt - loss_pivot) / beta

            #* param update
            w_new = w - ak * ghat
            torch.nn.utils.vector_to_parameters(w_new, self.model.program.parameters())
            
            loss = loss_pivot
            acc = compute_accuracy(output, label)[0].item()
        loss_summary = {"loss": loss,"acc": acc,}
        return loss_summary

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        self.total_length = self.num_batches * self.max_epoch
        self.warmup_length = self.total_length * 0.1

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            self.step += 1
            self.step_for_pdecay += 1
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
                    model_name="model-best.pth.tar"
                )

        if self.cfg.TRAINER.BASIC.MAX_API_CALLS and self.API_calls >= self.cfg.TRAINER.BASIC.MAX_API_CALLS:
            print("Reach the maximum number of API calls")
            self.epoch = self.max_epoch - 1
            last_epoch = True

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
            
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
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
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

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if self.cfg.TRAINER.BASIC.MAX_API_CALLS and self.API_calls >= self.cfg.TRAINER.BASIC.MAX_API_CALLS:
                break
        self.after_train()