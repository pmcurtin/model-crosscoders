from torch.nn import Linear
from .models import (
    CrossCoder,
    BetterCrossCoder,
    ResidualBuffer,
    SAE,
    SAEBuffer,
    LinearFeatureTransfer,
)
from .config import default_cfg
from torch.nn.utils import clip_grad_norm_

import torch
import wandb
import tqdm

# https://github.com/neelnanda-io/Crosscoders/blob/3adc7eb23a5a56f12557d2c6c206f0aa688bdbd3/crosscoders/utils.py#L500


# some trainer class.
# we should use weights & biases for experiment tracking
class CrossCoderTrainer:
    def __init__(
        self,
        modelA,
        modelB,
        dataloader,
        use_qwen=True,
        use_wandb=True,
        use_better=False,
        linear=False,
        cfg=default_cfg,
    ):
        self.cfg = cfg
        self.total_steps = self.cfg["num_tokens"] // self.cfg["batch_size"]
        self.step_counter = 0
        self.better = use_better
        if linear:
            self.crosscoder = LinearFeatureTransfer(
                self.cfg, model_dim_a=modelA.cfg.d_model, model_dim_b=modelB.cfg.d_model
            ).to(self.cfg["device"])
        else:
            if use_better:
                self.crosscoder = BetterCrossCoder(
                    self.cfg,
                    model_dim_a=modelA.cfg.d_model,
                    model_dim_b=modelB.cfg.d_model,
                ).to(self.cfg["device"])
            else:
                self.crosscoder = CrossCoder(
                    self.cfg,
                    model_dim_a=modelA.cfg.d_model,
                    model_dim_b=modelB.cfg.d_model,
                ).to(self.cfg["device"])
        self.buffer = ResidualBuffer(self.cfg, modelA, modelB, dataloader, use_qwen)

        self.optimizer = torch.optim.AdamW(
            self.crosscoder.parameters(),
            lr=self.cfg["lr"],
            betas=(self.cfg["beta1"], self.cfg["beta2"]),
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )

        if use_wandb:
            wandb.init(
                project="crosscoder",
                entity="cs2222-crosscoders",
                config=self.cfg,
            )

    def lr_lambda(self, step):
        if step < 0.05 * self.total_steps:
            return step / (0.05 * self.total_steps)
        elif step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def step(self):
        acts_a, acts_b = self.buffer.next()
        losses = self.crosscoder.losses(acts_a, acts_b)

        loss = losses[0]

        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2": losses[0].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

        if self.better:
            loss_dict.update(
                {
                    "aa": losses[1].item(),
                    "ab": losses[2].item(),
                    "ba": losses[3].item(),
                    "bb": losses[4].item(),
                }
            )

        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.crosscoder.save()

    def train(self):
        self.step_counter = 0
        
        try:
            for i in tqdm.trange(self.total_steps, desc="Training..."):
                loss_dict = self.step()
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            self.save()


class SAETrainer:
    def __init__(self, model, dataloader, use_wandb=True, cfg=default_cfg):
        self.cfg = cfg
        self.total_steps = self.cfg["num_tokens"] // self.cfg["batch_size"]
        self.step_counter = 0
        self.SAE = SAE(self.cfg, model_dim=model.cfg.d_model).to(self.cfg["device"])
        self.buffer = SAEBuffer(self.cfg, model, dataloader)

        self.optimizer = torch.optim.AdamW(
            self.SAE.parameters(),
            lr=self.cfg["lr"],
            betas=(self.cfg["beta1"], self.cfg["beta2"]),
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )

        if use_wandb:
            wandb.init(
                project="crosscoder",
                entity="cs2222-crosscoders",
                config=self.cfg,
            )

    def lr_lambda(self, step):
        if step < 0.05 * self.total_steps:
            return step / (0.05 * self.total_steps)
        elif step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

    def step(self):
        acts = self.buffer.next()
        losses = self.SAE.losses(acts)

        loss = losses[0]

        loss.backward()
        clip_grad_norm_(
            self.SAE.parameters(), max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2": losses[0].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.SAE.save()

    def train(self):
        self.step_counter = 0
        
        try:
            for i in tqdm.trange(self.total_steps, desc="Training..."):
                loss_dict = self.step()
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            self.save()
