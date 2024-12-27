import os.path as osp
import torch
import numpy as np
import time, datetime
import torch.nn as nn
import random
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import (
    MetricMeter, AverageMeter, load_checkpoint, load_pretrained_weights
)
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from projections.intrinsic import fastfood_vars, random_vars, fastfood_torched, random_torched
from torchinfo import summary
from trainers.utils import load_clip_to_cpu
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.cfg = cfg

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x        


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.ZIP.N_CTX
        ctx_init = cfg.TRAINER.ZIP.CTX_INIT
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.dtype = clip_model.dtype
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx.requires_grad = True
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts


class IntrinsicLearner(nn.Module):
    def __init__(self, cfg, prompt_learner: nn.Module, device):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.dtype = prompt_learner.dtype
        self.ID = cfg.TRAINER.ZIP.INTRINSIC_DIM
        self.rank = cfg.TRAINER.ZIP.RANK

        self.m = [prompt_learner]
        self.name_base_localname = []
        self.initial_value = dict()
        self.fastfood_params = {}

        # iteratre over layers in the module
        layers = 0
        for name, param in prompt_learner.named_parameters():
            # if param requires grad update
            if param.requires_grad:
                self.nctx, self.dim = param.size()
                for i in range(self.nctx):
                    layers += 1
                    ctx_name = name + f"_{i}"
                    self.initial_value[ctx_name] = v0 = (
                        param[i].clone().detach().requires_grad_(False).to(device)
                    )
                    # generate fastfood parameters
                    DD = torch.prod(torch.tensor(v0.size()))
                    self.fastfood_params[ctx_name] = fastfood_vars(DD, device)
                    base, localname = prompt_learner, ctx_name
                    while "." in localname:
                        prefix, localname = localname.split(".", 1)
                        base = base.__getattr__(prefix)
                    self.name_base_localname.append((name, base, localname))
        for name, base, localname in self.name_base_localname:
            delattr(base, name)
            break

        # parameter vector that is updated
        self.U = nn.Parameter(torch.zeros((layers, self.rank)).to(device))
        self.s = nn.Parameter(torch.ones((self.rank)).to(device))
        self.V = nn.Parameter(torch.randn((self.ID//layers, self.rank)).to(device))
        self.shared_V = nn.Parameter(torch.zeros((self.ID//layers)).to(device))

        self.register_parameter("U", self.U)
        self.register_parameter("s", self.s)
        self.register_parameter("V", self.V)
        self.register_parameter("shared_V", self.shared_V)

    def forward(self):
        index = 0
        reconstruction_matrix = (self.U @ torch.diag(self.s) @ self.V.t() + self.shared_V.unsqueeze(0)).to(self.device)

        # iterate over layers
        for name, base, localname in self.name_base_localname:
            set_params = torch.empty(self.nctx, self.dim, dtype=self.dtype).to(self.device)
            for i in range(self.nctx):
                ctx_name = name + f"_{i}"
                init_shape = self.initial_value[ctx_name].size()
                DD = torch.prod(torch.tensor(init_shape))
                ray = fastfood_torched(reconstruction_matrix[i], DD, self.fastfood_params[ctx_name]).view(init_shape)
                param = self.initial_value[ctx_name] + ray
                if self.cfg.TRAINER.ZIP.PREC == "fp16":
                    param = param.type(torch.float16)
                set_params[i] = param
            setattr(base, name, set_params)

        # pass image through the model, by getting hte module from a list self.m
        original_module = self.m[0].to(self.device)
        prompts = original_module()

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.intrinsic_learner = IntrinsicLearner(cfg, prompt_learner, device)
        self.tokenized_prompts = prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.intrinsic_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class ZIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ZIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.ZIP.PREC == "fp32" or cfg.TRAINER.ZIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        #! blackbox setting
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.intrinsic_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.intrinsic_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("intrinsic_learner", self.model.intrinsic_learner, self.optim, self.sched)
        self.N_params = len(torch.nn.utils.parameters_to_vector(self.model.intrinsic_learner.parameters()))

        # print model.intrinsic_learner summary
        for name, param in self.model.intrinsic_learner.named_parameters():
            print(name, param.size())

        print(f"Number of optimization variables: {self.N_params}")

        #! ZIP parameters
        self.opt_type                                  = cfg.TRAINER.BASIC.OPT_TYPE
        self.sp_avg                                    = cfg.TRAINER.BASIC.SP_AVG
        self.o, self.c, self.a, self.alpha, self.gamma = cfg.TRAINER.ZIP.SPSA_PARAMS
        self.b1                                        = cfg.TRAINER.ZIP.MOMS

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
                w = torch.nn.utils.parameters_to_vector(self.model.intrinsic_learner.parameters())
                ghat, loss, acc = self.spsa_grad_estimate_bi(self.sp_avg, w, image, label, ck)

                if self.opt_type == 'spsa-gc':
                    if self.step > 1:  self.m1 = self.b1*self.m1 + ghat
                    else:              self.m1 = ghat
                    accum_ghat = ghat + self.b1*self.m1
                elif self.opt_type == 'spsa':
                    accum_ghat = ghat
                elif self.opt_type == 'spsa-clip':
                    NORM = torch.norm(ghat).item()
                    DELTA = np.sqrt(len(w))
                    if NORM >= DELTA:
                        accum_ghat = (DELTA / NORM) * ghat
                    else:
                        accum_ghat = ghat
                else:
                    raise ValueError

                #* param update
                w_new = (w - ak * accum_ghat).to(w.dtype)
                torch.nn.utils.vector_to_parameters(w_new, self.model.intrinsic_learner.parameters()) 

        loss_summary = {"loss": loss, "acc": acc,}
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
            w_r = (w + ck*perturb).to(w.dtype)
            w_l = (w - ck*perturb).to(w.dtype)

            torch.nn.utils.vector_to_parameters(w_r, self.model.intrinsic_learner.parameters())
            output1 = self.model(image)
            torch.nn.utils.vector_to_parameters(w_l, self.model.intrinsic_learner.parameters())
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