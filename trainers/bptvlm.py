import os.path as osp
import time, datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchinfo import summary
from es_algorithms.CMA_ES import shallow_cma
from trainers.utils import load_clip_to_cpu
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model, classnames, device):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.device = device

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BPTVLM.TOKENS_L
        self.dtype = clip_model.dtype
        self.n_prompt_tokens_L = cfg.TRAINER.BPTVLM.TOKENS_L
        self.intrinsic_dim_L = cfg.TRAINER.BPTVLM.INTRINSIC_DIM_L

        # Text Encoder
        prompt_prefix = " ".join(["X"] * self.n_prompt_tokens_L)
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
        self.embedding = embedding.to(device)
        self.name_lens = name_lens

    def forward(self, prompts):
        x = self.incorporate_prompt(prompts, self.embedding[:self.n_cls])
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts[:self.n_cls].argmax(dim=-1)] @ self.text_projection
        return x
    
    def incorporate_prompt(self, prompt, embedding):
        if prompt == None:
            prefix = embedding[:, :1, :]
            suffix = embedding[:, 1:, :]
            x = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return x
        
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_prompt_tokens_L:, :]
        
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0).expand(self.n_cls, -1, -1)
        x = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                prompt,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return x

class VisionEncoder(nn.Module):
    def __init__(self, cfg, clip_model, device):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.n_prompt_tokens_V = cfg.TRAINER.BPTVLM.TOKENS_V

        # vision encoder variables
        self.input_resolution = clip_model.visual.input_resolution
        self.patch_size = clip_model.visual.patch_size
        self.prefix_len = (self.input_resolution//self.patch_size)**2+1
        self.output_dim = clip_model.visual.output_dim
        self.conv1 = clip_model.visual.conv1
        self.width = clip_model.visual.width
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.tranformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

    def forward(self, x, prompt):
        x = self.conv1(x)  # serial: (batch_size, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # shape = [*, grid ** 2 + 1, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + 
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
            ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        # Incorporate prompt
        x = self.incorporate_prompt(prompt, x)
        x = self.ln_pre(x)

        # Transformer
        x = x.permute(1,0,2)
        x = self.tranformer(x)
        x = x.permute(1,0,2)
        x = self.ln_post(x[:,0,:])

        if self.proj is not None:
            x = x @ self.proj
        return x

    def incorporate_prompt(self, prompt, embedding):
        B = embedding.shape[0]
        if prompt == None:
            return embedding
        
        # after CLS token, all before image patches
        embedding = torch.cat((
            embedding[:,:self.prefix_len,:],
            prompt.expand(B,-1,-1),
        ),dim=1)
        # [batch_size,cls_token + n_prompts_V + n_patches, hidden_dim]

        return embedding


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.text_encoder = TextEncoder(cfg, clip_model, classnames, device)
        self.vision_encoder = VisionEncoder(cfg, clip_model, device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device

        # Lauguage Linear Layer
        self.init_prompt = None
        self.intrinsic_dim_L = cfg.TRAINER.BPTVLM.INTRINSIC_DIM_L
        self.n_prompt_tokens_L = cfg.TRAINER.BPTVLM.TOKENS_L
        self.ctx_dim_L = clip_model.ln_final.weight.shape[0]
        self.sigma = cfg.TRAINER.BPTVLM.SIGMA
        self.linear_L = torch.nn.Linear(
            self.intrinsic_dim_L, 
            self.n_prompt_tokens_L * self.ctx_dim_L,
            bias=False, device=self.device, dtype=self.dtype
            )
        embedding = clip_model.token_embedding.weight.cpu()
        mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
        mu = 0.0
        std = std_hat / (np.sqrt(self.intrinsic_dim_L) * self.sigma)

        print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_L.parameters():
            torch.nn.init.normal_(p, mu, std)

        # Vision Linear Layer
        self.intrinsic_dim_V = cfg.TRAINER.BPTVLM.INTRINSIC_DIM_V
        self.n_prompt_tokens_V = cfg.TRAINER.BPTVLM.TOKENS_V
        self.ctx_dim_V = clip_model.visual.width
        self.linear_V = torch.nn.Linear(
            self.intrinsic_dim_V, 
            self.n_prompt_tokens_V * self.ctx_dim_V, 
            bias=False, device=self.device, dtype=self.dtype
            )
        conv = clip_model.visual.conv1.weight.cpu()
        mu_hat = np.mean(conv.reshape(-1).detach().cpu().numpy())
        std_hat = np.std(conv.reshape(-1).detach().cpu().numpy())
        mu = mu_hat * 3072 / self.intrinsic_dim_V
        std = std_hat * np.sqrt(3072 / self.intrinsic_dim_V) * self.sigma

        print('[Conv] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
        for p in self.linear_V.parameters():
            torch.nn.init.normal_(p, mu, std)

    def forward(self, image, intrinsic_vector, prompt_zips):
        # up project intrinsic vector
        if intrinsic_vector is not None:
            self.text_prompts = self.generate_text_prompts(intrinsic_vector[:self.intrinsic_dim_L])
            self.image_prompts = self.generate_visual_prompts(intrinsic_vector[self.intrinsic_dim_L:])
        else :
            self.text_prompts, self.image_prompts = prompt_zips[0], prompt_zips[1]

        image = image.type(self.dtype)
        image_features = self.vision_encoder(image, self.image_prompts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_encoder(self.text_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

    def generate_text_prompts(self, intrinsic_vector):
        z = torch.tensor(intrinsic_vector, device=self.device, dtype=self.dtype)
        # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
        z = self.linear_L(z).reshape(self.n_prompt_tokens_L, -1)
        if self.init_prompt is not None:
            z = z + self.init_prompt  # Az + p_0
        return z
    
    def generate_visual_prompts(self, intrinsic_vector):
        z = torch.tensor(intrinsic_vector, device=self.device, dtype=self.dtype)
        # [intrinsic_dim_L,] -> [n_prompt_token,ctx_dim]
        z = self.linear_V(z).reshape(self.n_prompt_tokens_V, -1)
        return z


@TRAINER_REGISTRY.register()
class BPTVLM(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BPTVLM.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.BPTVLM.PREC == "fp32" or cfg.TRAINER.BPTVLM.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        #! blackbox setting
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        self.model.to(self.device)
        self.register_model("text_encoder", self.model.text_encoder)
        self.step = 0
        self.loss_fn = F.cross_entropy

        #! BPT-VLM
        self.ndim_problem = cfg.TRAINER.BPTVLM.INTRINSIC_DIM_L + cfg.TRAINER.BPTVLM.INTRINSIC_DIM_V
        self.intrinsic_vector = 0 * np.ones((self.ndim_problem,))
        self.min_loss = None
        self.best_prompt_text = None
        self.best_prompt_image = None

        self.opt_cfg = {'fitness_threshold': 1e-10,
           'seed_rng': 0,
           'max_runtime': 20800,
           'x': self.intrinsic_vector,  # mean
           'sigma': cfg.TRAINER.BPTVLM.SIGMA,
           'verbose_frequency': 5,
           'n_individuals': cfg.TRAINER.BPTVLM.POP_SIZE,
           'is_restart': False}
        self.opt = shallow_cma(cfg)
        self.API_calls = 0

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch, intrinsic_vector):
        with torch.no_grad():
            image, label = self.parse_batch_train(batch)
            output = self.model(image, intrinsic_vector, None)
            self.API_calls += 1
            loss = self.loss_fn(output, label)
            acc = compute_accuracy(output, label)[0].item()
        loss_summary = {"loss": loss.item(), "acc": acc, }
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def train(self):
        self.time_start = time.time()
        API_STOP = False
        while self.epoch < self.max_epoch:
            self.before_epoch()
            solutions = self.opt.ask()
            x_list = [x for x in solutions]
            fitnesses = []
            for x in x_list:
                fitnesses.append(self.run_epoch(x))
                self.after_epoch()
                if self.cfg.TRAINER.BASIC.MAX_API_CALLS and self.API_calls >= self.cfg.TRAINER.BASIC.MAX_API_CALLS:
                    API_STOP = True
                    break
                self.epoch += 1
            if API_STOP:
                break
            self.opt.tell(solutions, fitnesses)
        self.after_train()

    def run_epoch(self, intrinsic_vector):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            self.step += 1
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, intrinsic_vector)
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
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/", n_iter)
            end = time.time()

            #* added part
            epoch_loss = losses.meters['loss'].avg
            if self.min_loss is None or epoch_loss < self.min_loss : 
                self.min_loss = epoch_loss
                self.best_prompt_text = self.model.text_prompts
                self.best_prompt_image = self.model.image_prompts

            if self.cfg.TRAINER.BASIC.MAX_API_CALLS and self.API_calls >= self.cfg.TRAINER.BASIC.MAX_API_CALLS:
                break

        #* added part
        print(self.API_calls)
        epoch_acc = losses.meters['acc'].avg
        epoch_loss = losses.meters['loss'].avg
        return epoch_loss

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

    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            with torch.no_grad():
                output = self.model(input, None, (self.best_prompt_text, self.best_prompt_image))
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name="", prompts=None
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result,
                    "text_prompts": prompts[0],
                    "image_prompts": prompts[1],
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

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
            self.best_prompt_text = checkpoint["text_prompts"]
            self.best_prompt_image = checkpoint["image_prompts"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    prompts = (self.model.text_prompts, self.model.image_prompts),
                    model_name="model-best.pth.tar"
                )

        if self.cfg.TRAINER.BASIC.MAX_API_CALLS and self.API_calls >= self.cfg.TRAINER.BASIC.MAX_API_CALLS:
            print("Reach the maximum number of API calls")
            self.epoch = self.max_epoch - 1
            last_epoch = True

        if meet_checkpoint_freq or last_epoch:
            self.save_model(
                epoch = self.epoch, 
                directory = self.output_dir, 
                prompts = (self.model.text_prompts, self.model.image_prompts)
            )