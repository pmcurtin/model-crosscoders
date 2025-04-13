from __future__ import annotations

import argparse
import torch
from torch import nn
import torch.nn.functional as F

import einops

from typing import Any
from pathlib import Path

import transformer_lens
import tqdm

from config import default_cfg

# cfg = arg_parse_update_cfg(default_cfg)
cfg = default_cfg  # placeholder for now since we're lazy


class CrossCoder(nn.Module):
    def __init__(
        self,
        cfg,
        modelA: transformer_lens.HookedTransformer,
        modelB: transformer_lens.HookedTransformer,
    ):
        super().__init__()
        # modelA and modelB are imported because presumably we use them to get resid sizes

        # TODO: we don't need to be passing the entire models to a crosscoder
        # we just need the residual dimensions

        self.cfg: dict[str, Any] = cfg  # config

        self.save_dir = Path(self.cfg["save_dir"])
        self.save_version: int = self.cfg["save_version"]

        self.hidden_dim = int(self.cfg["dict_size"])  # pyright: ignore

        self.topk = 128  # idk

        self.encoders = nn.ModuleList(
            [
                nn.Linear(modelA.cfg.d_model, self.hidden_dim, dtype=torch.bfloat16),
                nn.Linear(modelB.cfg.d_model, self.hidden_dim, dtype=torch.bfloat16),
            ]
        )
        self.decoders = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, modelA.cfg.d_model, dtype=torch.bfloat16),
                nn.Linear(self.hidden_dim, modelB.cfg.d_model, dtype=torch.bfloat16),
            ]
        )

        pass

    def encode(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """
        x: input residual
        m: which model (0, or 1) the residual's from

        returns: encoded latents
        """
        # has two encoders, so m indicates which to use

        raw_enc: torch.Tensor = self.encoders[m](x)  # pyright: ignore
        return self.topk_constraint(raw_enc)  # pyright: ignore

    def decode(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """
        x: latents
        m: which model to decode to

        returns: decoded (reconstructed) residuals for model m
        """
        return self.decoders[m](x)

    def topk_constraint(self, x: torch.Tensor) -> torch.Tensor:
        # DOES THIS WORK?
        # get topk
        # zeroed_idxs = torch.argsort(x, dim=1)[:, : -self.topk]
        # print(zeroed_idxs)
        # x[zeroed_idxs] = 0

        return x.scatter(index=torch.argsort(x, dim=1)[:, : -self.topk], dim=1, value=0)

    def forward(self, x: torch.Tensor, encode_m: int, decode_m: int) -> torch.Tensor:
        """
        x: input residual
        encode_m: which model encoder to use
        decode_m: which model decoder to use

        returns: reconstructed residual of model `decode_m`
        """

        encoded: torch.Tensor = self.encoders[encode_m](x)  # pyright: ignore
        sparse_encoded = self.topk_constraint(encoded)  # pyright: ignore
        decoded = self.decoders[decode_m](sparse_encoded)

        return decoded

    def losses(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        x_a: residuals from model a
        x_b: residuals from model b

        returns: mse losses of all four training objectives
        """

        # compute loss.
        # losses from: (a -> a, a -> b, b -> a, b -> b) reconstructions
        aa = self.forward(x_a, 0, 0)
        ab = self.forward(x_a, 0, 1)
        ba = self.forward(x_b, 1, 0)
        bb = self.forward(x_b, 1, 1)

        return (
            F.mse_loss(aa, x_a),
            F.mse_loss(ab, x_b),
            F.mse_loss(ba, x_a),
            F.mse_loss(bb, x_b),
        )

    # folling functions straight from neel
    def create_save_dir(self):
        version_list = [
            int(file.name.split("_")[1])
            for file in list(self.save_dir.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.version_save_dir = self.save_dir / f"version_{version}"
        self.version_save_dir.mkdir(parents=True)

    def save(self):
        if self.version_save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1

    @classmethod
    def load(
        cls,
        name,
        model=None,
        path="",
    ):
        # If the files are not in the default save directory, you can specify a path
        # It's assumed that weights are [name].pt and cfg is [name]_cfg.json
        if path == "":
            save_dir = self.save_dir
        else:
            save_dir = Path(path)
        cfg_path = save_dir / f"{str(name)}_cfg.json"
        weight_path = save_dir / f"{str(name)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        # pprint.pprint(cfg)
        if model is None:
            model = (
                HookedTransformer.from_pretrained(cfg["model_name"])
                .to(DTYPES[cfg["enc_dtype"]])
                .to(cfg["device"])
            )
        self = cls(cfg=cfg, model=model)
        self.load_state_dict(torch.load(weight_path))
        return self


# I like Neel Nanda's buffer method for generating residuals on-the-fly
# read his impl first. Ours will be simpler probably
class ResidualBuffer:
    def __init__(
        self,
        cfg: dict,
        model_a: transformer_lens.HookedTransformer,
        model_b: transformer_lens.HookedTransformer,
        dataloader_a: torch.utils.data.DataLoader,
        dataloader_b: torch.utils.data.DataLoader,
    ):
        """
        cfg: config file
        model: model generating residuals stored in this buffer
        """

        self.cfg = cfg
        self.model_a = model_a
        self.model_b = model_b
        self.dataloader_a_iter = iter(dataloader_a)
        self.dataloader_b_iter = iter(dataloader_b)

        self.buffer_size: int = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches: int = self.buffer_size // (
            (cfg["seq_len"] - 1) * cfg["model_batch_size"]
        )
        self.buffer_size: int = (
            self.buffer_batches * (cfg["seq_len"] - 1) * cfg["model_batch_size"]
        )  # clip to exact multiple of seq_len minus BOS

        # print(cfg["batch_size"], cfg["buffer_mult"], cfg["seq_len"])
        # print(self.buffer_size, self.buffer_batches)

        self.buffer_a = self.init_buffer(self.model_a)
        self.buffer_b = self.init_buffer(self.model_b)

        self.first_fill = True  # whether first call to refresh has occurred
        self.pointer = 0

        # normalization factor??

        self.refresh()

    @torch.no_grad()
    def init_buffer(self, model: transformer_lens.HookedTransformer) -> torch.Tensor:
        return torch.zeros(
            (self.buffer_size, model.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
            device=cfg["device"],
        )

    @torch.no_grad()
    def refresh(self):
        # initally fill the buffer (first call)
        # then just discard old examples and top it up on subsequent calls
        # we can use transformerlens here (easy, use activation cache), or some fancier way of getting
        # the residuals that we want (for example, no need to execute transformer past
        # layer of interest)
        self.pointer = 0
        num_batches = (
            self.buffer_batches // 2 if not self.first_fill else self.buffer_batches
        )
        self.first_fill = False

        print(num_batches, self.buffer_batches, self.buffer_size)

        for i in tqdm.trange(0, num_batches, desc="Filling buffer..."):
            batch_a = next(self.dataloader_a_iter)["input_ids"]
            batch_b = next(self.dataloader_b_iter)["input_ids"]

            _, cache = self.model_a.run_with_cache(
                batch_a, names_filter=lambda x: x.endswith("resid_post")
            )
            cache: transformer_lens.ActivationCache

            LAYER = 4

            acts_a = einops.rearrange(
                cache.stack_activation("resid_post")[LAYER, :, 1:, :],
                "batch seq_len d_model -> (batch seq_len) d_model",
            )

            _, cache = self.model_b.run_with_cache(
                batch_b, names_filter=lambda x: x.endswith("resid_post")
            )
            cache: transformer_lens.ActivationCache

            acts_b = einops.rearrange(
                cache.stack_activation("resid_post")[LAYER, :, 1:, :],
                "batch seq_len d_model -> (batch seq_len) d_model",
            )

            self.buffer_a[self.pointer : self.pointer + acts_a.shape[0]] = acts_a
            self.buffer_b[self.pointer : self.pointer + acts_b.shape[0]] = acts_b
            self.pointer += acts_a.shape[0]

        self.pointer = 0
        self.buffer_a = self.buffer_a[
            torch.randperm(self.buffer_a.shape[0]).to(self.cfg["device"])
        ]
        self.buffer_b = self.buffer_b[
            torch.randperm(self.buffer_b.shape[0]).to(self.cfg["device"])
        ]

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        end: int = self.pointer + self.cfg["batch_size"]
        activations_a = self.buffer_a[self.pointer : end]
        activations_b = self.buffer_b[self.pointer : end]
        self.pointer = end

        if self.pointer > self.buffer_size // 2:
            self.refresh()

        # normalize??

        return activations_a, activations_b


class ModelWithEncoder(nn.Module):
    # I think delphi needs a model that takes in text and spits out
    # SAE features. This can be that model

    def __init__(self, model, encoder, layer):
        pass

    def forward(self, x):
        # first run model on x
        # then run encoder on residual
        pass
