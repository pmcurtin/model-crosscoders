from models import CrossCoder, ResidualBuffer
from config import default_cfg

import torch
import wandb
import tqdm

# https://github.com/neelnanda-io/Crosscoders/blob/3adc7eb23a5a56f12557d2c6c206f0aa688bdbd3/crosscoders/utils.py#L500


# some trainer class.
# we should use weights & biases for experiment tracking
class CrossCoderTrainer:
    def __init__(self, modelA, modelB, dataloader_a, dataloader_b, use_wandb=True):
        self.cfg = default_cfg  # populate this from args
        self.total_steps = self.cfg["num_tokens"] // self.cfg["batch_size"]
        self.step_counter = 0

        self.crosscoder = CrossCoder(self.cfg, modelA, modelB)
        self.buffer = ResidualBuffer(self.cfg, modelA, modelB, dataloader_a, dataloader_b)

        self.optimizer = torch.optim.AdamW(
            self.crosscoder.parameters(),
            lr=self.cfg["lr"],
            betas=(self.cfg["beta1"], self.cfg["beta2"]),
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )

        if use_wandb:
            wandb.init(project="crosscoder", entity=WANDB_PROJECT, config=default_cfg)

    def lr_lambda(self, step):
        if step < 0.05 * self.total_steps:
            return step / (0.05 * self.total_steps)
        elif step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def step(self):
        acts_a, acts_b = self.buffer.next()
        losses = self.crosscoder.losses(acts_a, acts_b)

        loss = torch.mean(torch.cat(losses, dim=0), dim=0)

        loss.backward()
        # TODO: do we need to do this "clip grad norm" shit
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "aa_loss": losses[0].item(),
            "ab_loss": losses[1].item(),
            "ba_loss": losses[2].item(),
            "bb_loss": losses[3].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    # def save(self):
    #     self.crosscoder.save()

    def train(self):
        self.step_counter = 0
        try:
            for i in tqdm.trange(self.total_steps):
                loss_dict = self.step()
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                # if (i + 1) % self.cfg["save_every"] == 0:
                #     self.save()
        finally:
            self.save()
