from __future__ import annotations

import argparse
import torch
from torch import nn
import torch.nn.functional as F
import json
import numpy as np

import einops

from typing import Any
from pathlib import Path

import transformer_lens
from transformer_lens.utils import get_act_name
import tqdm

from .config import default_cfg

cfg = default_cfg  # placeholder for now since we're lazy


class CrossCoder(nn.Module):
    def __init__(
        self,
        cfg,
        model_dim_a: int,
        model_dim_b: int,
    ):
        super().__init__()
        # TODO: we don't need to be passing the entire models to a crosscoder
        # we just need the residual dimensions

        self.cfg: dict[str, Any] = cfg  # config

        self.save_dir = Path(self.cfg["save_dir"])
        self.save_version: int = self.cfg["save_version"]
        self.version_save_dir = None

        self.hidden_dim = int(self.cfg["dict_size"])  # pyright: ignore

        self.topk = self.cfg["topk"]  # idk

        self.encoder = nn.Parameter(
            torch.empty(
                model_dim_a + model_dim_b,
                self.hidden_dim,
                dtype=torch.bfloat16,
            )
        )

        self.encoder_bias = nn.Parameter(
            torch.zeros(self.hidden_dim, dtype=torch.bfloat16)
        )

        self.decoder = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    self.hidden_dim,
                    model_dim_a + model_dim_b,
                    dtype=torch.bfloat16,
                )
            )
        )

        self.decoder_bias = nn.Parameter(
            torch.zeros(
                model_dim_a + model_dim_b,
                dtype=torch.bfloat16,
            )
        )

        self.decoder.data = (
            self.decoder.data / self.decoder.data.norm(dim=-1, keepdim=True)
        ) * self.cfg["dec_init_norm"]
        self.encoder.data = einops.rearrange(
            self.decoder.data.clone(),
            "d_hidden d_model -> d_model d_hidden",
        )

        # self.encoder_a = nn.Parameter(
        #     torch.empty(modelA.cfg.d_model, self.hidden_dim, dtype=torch.bfloat16)
        # )
        # self.encoder_b = nn.Parameter(
        #     torch.empty(modelB.cfg.d_model, self.hidden_dim, dtype=torch.bfloat16)
        # )
        # self.decoder_a = nn.Parameter(
        #     torch.nn.init.normal_(
        #         torch.empty(self.hidden_dim, modelA.cfg.d_model, dtype=torch.bfloat16)
        #     )
        # )
        # self.decoder_a.data = (
        #     self.decoder_a.data
        #     / self.decoder_a.data.norm(dim=-1, keepdim=True)
        #     * self.cfg["dec_init_norm"]
        # )
        # self.encoder_a.data = einops.rearrange(
        #     self.decoder_a.data.clone(),
        #     "d_hidden d_model -> d_model d_hidden",
        # )
        # self.decoder_b = nn.Parameter(
        #     torch.nn.init.normal_(
        #         torch.empty(self.hidden_dim, modelB.cfg.d_model, dtype=torch.bfloat16)
        #     )
        # )
        # self.decoder_b.data = (
        #     self.decoder_b.data
        #     / self.decoder_b.data.norm(dim=-1, keepdim=True)
        #     * self.cfg["dec_init_norm"]
        # )
        # self.encoder_b.data = einops.rearrange(
        #     self.decoder_b.data.clone(),
        #     "d_hidden d_model -> d_model d_hidden",
        # )

        # self.b_encoder_a = nn.Parameter(
        #     torch.zeros(self.hidden_dim, dtype=torch.bfloat16)
        # )
        # self.b_encoder_b = nn.Parameter(
        #     torch.zeros(self.hidden_dim, dtype=torch.bfloat16)
        # )
        # self.b_decoder_a = nn.Parameter(
        #     torch.zeros(modelA.cfg.d_model, dtype=torch.bfloat16)
        # )
        # self.b_decoder_b = nn.Parameter(
        #     torch.zeros(modelB.cfg.d_model, dtype=torch.bfloat16)
        # )

        # self.encoders = nn.ModuleList(
        #     [
        #         nn.Linear(modelA.cfg.d_model, self.hidden_dim, dtype=torch.bfloat16),
        #         nn.Linear(modelB.cfg.d_model, self.hidden_dim, dtype=torch.bfloat16),
        #     ]
        # )
        # self.decoders = nn.ModuleList(
        #     [
        #         nn.Linear(self.hidden_dim, modelA.cfg.d_model, dtype=torch.bfloat16),
        #         nn.Linear(self.hidden_dim, modelB.cfg.d_model, dtype=torch.bfloat16),
        #     ]
        # )

        # print(self.encoder)
        # print(self.decoder)

        self.act = F.relu

    def normalize(
        self, x: torch.Tensor, eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)

        return x, mu, std

    def encode(
        self,
        x: torch.Tensor,  # m: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: input residual
        m: which model (0, or 1) the residual's from

        returns: encoded latents
        """

        return self.act(self.topk_constraint(x @ self.encoder + self.encoder_bias))

        # has two encoders, so m indicates which to use
        x, mu, std = self.normalize(x)
        raw_enc: torch.Tensor = (x @ (self.encoder_a, self.encoder_b)[m]) + (
            self.b_encoder_a,
            self.b_encoder_b,
        )[m]  # self.encoders[m](x)  # pyright: ignore
        # return self.act(self.topk_constraint(raw_enc)), mu, std  # pyright: ignore
        return self.act(raw_enc), mu, std

    def decode(
        self,
        x: torch.Tensor,  # , m: int, mu: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """
        x: latents
        m: which model to decode to

        returns: decoded (reconstructed) residuals for model m
        """

        return x @ self.decoder + self.decoder_bias

        return (
            x @ (self.decoder_a, self.decoder_b)[m]
            + (self.b_decoder_a, self.b_decoder_b)[m]
        ) * std + mu

    def topk_constraint(self, x: torch.Tensor) -> torch.Tensor:
        # DOES THIS WORK?
        # get topk
        # zeroed_idxs = torch.argsort(x, dim=1)[:, : -self.topk]
        # print(zeroed_idxs)
        # x[zeroed_idxs] = 0

        topk = torch.topk(x, k=self.topk, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, topk.values)
        return result

        # return x.scatter(index=torch.argsort(x, dim=1)[:, : -self.topk], dim=1, value=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # encode_m: int, decode_m: int
        """
        x: input residual
        encode_m: which model encoder to use
        decode_m: which model decoder to use

        returns: reconstructed residual of model `decode_m`
        """

        return self.decode(self.encode(x))

        latents, mu, std = self.encode(x, encode_m)  # pyright: ignore
        decoded = self.decode(latents, decode_m, mu, std)

        return decoded

    def losses(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        x_a: residuals from model a
        x_b: residuals from model b

        returns: mse losses of all four training objectives
        """

        # compute loss.
        # losses from: (a -> a, a -> b, b -> a, b -> b) reconstructions

        concat = torch.cat((x_a, x_b), dim=-1)

        acts = self.encode(concat)

        recon = self.decode(acts)  # self.forward(concat)

        # print(torch.sqrt(torch.sum(torch.square(concat - recon))))
        # print(concat)
        # print(recon)

        l2 = torch.sum(
            torch.square(concat - recon), dim=-1
        ).mean()  # F.mse_loss(concat, recon)

        dec_norm = self.decoder.norm(dim=-1)
        l1 = (acts @ dec_norm).mean()

        l0 = (acts > 0).float().sum(-1).mean()

        # acts_a, mu_a, std_a = self.encode(x_a, 0)
        # acts_b, mu_b, std_b = self.encode(x_b, 1)

        # aa = self.decode(acts_a, 0, mu_a, std_a)
        # ab = self.decode(acts_a, 1, mu_a, std_a)
        # ba = self.decode(acts_b, 0, mu_b, std_b)
        # bb = self.decode(acts_b, 1, mu_b, std_b)

        # dec_norm = self.decoder_a.norm(dim=-1) + self.decoder_b.norm(dim=-1)

        # l1 = ((acts_a + acts_b) @ dec_norm).mean(0)

        # l0 = (acts_a > 0).float().sum(-1).mean() + (acts_b > 0).float().sum(
        #     -1
        # ).mean() / 2

        return (
            # l2 losses on each pathway
            # F.mse_loss(aa, x_a),
            # F.mse_loss(ab, x_b),
            # F.mse_loss(ba, x_a),
            # F.mse_loss(bb, x_b),
            l2,
            l1,
            l0,
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
        weight_path = self.version_save_dir / f"{self.save_version}.pt"
        cfg_path = self.version_save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print(f"Saved as version {self.save_version} in {self.version_save_dir}")
        self.save_version += 1

    @classmethod
    def load(
        cls,
        name,
        model_dim_a,
        model_dim_b,
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
        self = cls(cfg=cfg, model_dim_a=model_dim_a, model_dim_b=model_dim_b)
        self.load_state_dict(torch.load(weight_path))
        return self


class BetterCrossCoder(nn.Module):
    def __init__(self, cfg, model_dim_a: int, model_dim_b: int):
        super().__init__()
        # TODO: we don't need to be passing the entire models to a crosscoder
        # we just need the residual dimensions

        self.cfg: dict[str, Any] = cfg  # config

        self.save_dir = Path(self.cfg["save_dir"])
        self.save_version: int = self.cfg["save_version"]
        self.version_save_dir = None

        self.hidden_dim = int(self.cfg["dict_size"])  # pyright: ignore

        self.topk = self.cfg["topk"]  # idk

        self.encoder_a = nn.Parameter(
            torch.empty(
                model_dim_a,
                self.hidden_dim,
                dtype=torch.bfloat16,
            )
        )

        self.encoder_a_bias = nn.Parameter(
            torch.zeros(self.hidden_dim, dtype=torch.bfloat16)
        )

        self.decoder_a = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    self.hidden_dim,
                    model_dim_a,
                    dtype=torch.bfloat16,
                )
            )
        )

        self.decoder_a_bias = nn.Parameter(
            torch.zeros(
                model_dim_a,
                dtype=torch.bfloat16,
            )
        )

        self.decoder_a.data = (
            self.decoder_a.data / self.decoder_a.data.norm(dim=-1, keepdim=True)
        ) * self.cfg["dec_init_norm"]
        self.encoder_a.data = einops.rearrange(
            self.decoder_a.data.clone(),
            "d_hidden d_model -> d_model d_hidden",
        )

        self.encoder_b = nn.Parameter(
            torch.empty(
                model_dim_b,
                self.hidden_dim,
                dtype=torch.bfloat16,
            )
        )

        self.encoder_b_bias = nn.Parameter(
            torch.zeros(self.hidden_dim, dtype=torch.bfloat16)
        )

        self.decoder_b = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    self.hidden_dim,
                    model_dim_b,
                    dtype=torch.bfloat16,
                )
            )
        )

        self.decoder_b_bias = nn.Parameter(
            torch.zeros(
                model_dim_b,
                dtype=torch.bfloat16,
            )
        )

        self.decoder_b.data = (
            self.decoder_b.data / self.decoder_b.data.norm(dim=-1, keepdim=True)
        ) * self.cfg["dec_init_norm"]
        self.encoder_b.data = einops.rearrange(
            self.decoder_b.data.clone(),
            "d_hidden d_model -> d_model d_hidden",
        )

        self.act = F.relu

    def encode(self, x: torch.Tensor, in_model: int) -> torch.Tensor:
        """
        x: input residual
        m: which model (0, or 1) the residual's from

        returns: encoded latents
        """

        if in_model == 0:
            x = self.topk_constraint(x @ self.encoder_a + self.encoder_a_bias)
        else:
            x = self.topk_constraint(x @ self.encoder_b + self.encoder_b_bias)

        return x

    def decode(self, x: torch.Tensor, out_model: int) -> torch.Tensor:
        """
        x: latents
        m: which model to decode to

        returns: decoded (reconstructed) residuals for model m
        """

        if out_model == 0:
            x = x @ self.decoder_a + self.decoder_a_bias
        else:
            x = x @ self.decoder_b + self.decoder_b_bias

        return x

    def topk_constraint(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.topk, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, topk.values)
        return result

    def forward(self, x: torch.Tensor, in_model: int, out_model: int) -> torch.Tensor:
        """
        x: input residual
        encode_m: which model encoder to use
        decode_m: which model decoder to use

        returns: reconstructed residual of model `decode_m`
        """

        return self.decode(self.encode(x, in_model), out_model)

    def losses(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        x_a: residuals from model a
        x_b: residuals from model b

        returns: mse losses of all four training objectives
        """

        recons = torch.stack(
            [
                self.forward(x_a, 0, 0),
                self.forward(x_a, 0, 1),
                self.forward(x_b, 1, 0),
                self.forward(x_b, 1, 1),
            ],
            dim=0,
        )

        targets = torch.stack([x_a, x_b, x_a, x_b], dim=0)

        l2 = torch.sum(torch.square(targets - recons), dim=-1).mean(dim=-1)

        aa, ab, ba, bb = l2

        l2 = l2.mean()

        return (l2, aa, ab, ba, bb)

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
        weight_path = self.version_save_dir / f"{self.save_version}.pt"
        cfg_path = self.version_save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print(f"Saved as version {self.save_version} in {self.version_save_dir}")
        self.save_version += 1

    @classmethod
    def load(
        cls,
        name,
        model_dim_a,
        model_dim_b,
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
        self = cls(cfg=cfg, model_dim_a=model_dim_a, model_dim_b=model_dim_b)
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
        dataloader: torch.utils.data.DataLoader,
        use_qwen: bool,
        # dataloader_b: torch.utils.data.DataLoader,
    ):
        """
        cfg: config file
        model: model generating residuals stored in this buffer
        """

        self.cfg = cfg
        self.model_a = model_a
        self.model_b = model_b
        self.dataloader_iter = iter(dataloader)
        # self.dataloader_b_iter = iter(dataloader_b)

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
        if use_qwen:
            self.res_layers = (18, 18)  # take from layer 18 for qwen
        else:
            self.res_layers = (
                11,
                23,
            )  # take from layers (11, 23) for pythia; this is ~2/3 way through 160m and 410m respectively

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

        # print(num_batches, self.buffer_batches, self.buffer_size)

        for i in tqdm.trange(0, num_batches, desc="Filling buffer..."):
            batch = next(self.dataloader_iter)  # ["input_ids"]
            # batch_b = next(self.dataloader_b_iter)["input_ids"]
            LAYER_A, LAYER_B = self.res_layers

            _, cache = self.model_a.run_with_cache(
                batch["input_ids_a"],
                names_filter=lambda x: x.endswith("resid_post"),
                stop_at_layer=LAYER_A + 1,
            )
            cache: transformer_lens.ActivationCache

            acts_a = einops.rearrange(
                cache[get_act_name("resid_post", LAYER_A)][
                    :, 1:
                ],  # cache.stack_activation("resid_post")[LAYER, :, 1:, :],
                "batch seq_len d_model -> (batch seq_len) d_model",
            )

            _, cache = self.model_b.run_with_cache(
                batch["input_ids_b"],
                names_filter=lambda x: x.endswith("resid_post"),
                stop_at_layer=LAYER_B + 1,
            )
            cache: transformer_lens.ActivationCache

            acts_b = einops.rearrange(
                cache[get_act_name("resid_post", LAYER_B)][
                    :, 1:
                ],  # cache.stack_activation("resid_post")[LAYER, :, 1:, :],
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

        if self.first_fill:
            self.scaling_a = (
                np.sqrt(self.model_a.cfg.d_model)
                / (self.buffer_a.norm(dim=-1).mean() + 1e-5).item()
            )
            self.scaling_b = (
                np.sqrt(self.model_b.cfg.d_model)
                / (self.buffer_b.norm(dim=-1).mean() + 1e-5).item()
            )

            # print(self.scaling_a, self.scaling_b)

        self.first_fill = False

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        end: int = self.pointer + self.cfg["batch_size"]
        activations_a = self.buffer_a[self.pointer : end]
        activations_b = self.buffer_b[self.pointer : end]
        self.pointer = end

        if self.pointer > self.buffer_size // 2:
            self.refresh()

        # normalize??

        return activations_a * self.scaling_a, activations_b * self.scaling_b


class SAE(nn.Module):
    def __init__(self, cfg: dict[str, Any], model_dim: int):
        super().__init__()
        self.cfg: dict[str, Any] = cfg
        self.save_dir = Path(self.cfg["save_dir"])
        self.save_version: int = self.cfg["save_version"]
        self.version_save_dir = None
        self.hidden_dim = int(self.cfg["dict_size"])
        self.topk = self.cfg["topk"]

        self.encoder = nn.Parameter(
            torch.empty(
                model_dim,
                self.hidden_dim,
                dtype=torch.bfloat16,
            )
        )
        self.encoder_bias = nn.Parameter(
            torch.zeros(self.hidden_dim, dtype=torch.bfloat16)
        )

        self.decoder = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    self.hidden_dim,
                    model_dim,
                    dtype=torch.bfloat16,
                )
            )
        )
        self.decoder_bias = nn.Parameter(
            torch.zeros(
                model_dim,
                dtype=torch.bfloat16,
            )
        )

        self.decoder.data = (
            self.decoder.data / self.decoder.data.norm(dim=-1, keepdim=True)
        ) * self.cfg["dec_init_norm"]
        self.encoder.data = einops.rearrange(
            self.decoder.data.clone(),
            "d_hidden d_model -> d_model d_hidden",
        )
        self.act = F.relu

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.act(self.topk_constraint(x @ self.encoder + self.encoder_bias))

    def decode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x @ self.decoder + self.decoder_bias

    def topk_constraint(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.topk, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, topk.values)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def losses(self, x: torch.Tensor):
        acts = self.encode(x)
        recon = self.decode(acts)

        l2 = torch.sum(torch.square(x - recon), dim=-1).mean()

        return (l2,)

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
        weight_path = self.version_save_dir / f"{self.save_version}.pt"
        cfg_path = self.version_save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        print(f"Saved as version {self.save_version} in {self.version_save_dir}")
        self.save_version += 1

    @classmethod
    def load(
        cls,
        name,
        model_dim,
        path="",
    ):
        # if the files are not in the default save directory, you can specify a path
        if path == "":
            save_dir = self.save_dir
        else:
            save_dir = Path(path)
        # assumed that weights are [name].pt and cfg is [name]_cfg.json
        cfg_path = save_dir / f"{str(name)}_cfg.json"
        weight_path = save_dir / f"{str(name)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        self = cls(cfg=cfg, model_dim=model_dim)
        self.load_state_dict(torch.load(weight_path))
        return self


class SAEBuffer:
    def __init__(
        self,
        cfg: dict,
        model: transformer_lens.HookedTransformer,
        dataloader: torch.utils.data.DataLoader,
    ):
        self.cfg = cfg
        self.model = model
        self.dataloader_iter = iter(dataloader)

        self.buffer_size: int = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches: int = self.buffer_size // (
            (cfg["seq_len"] - 1) * cfg["model_batch_size"]
        )

        self.buffer_size: int = (
            self.buffer_batches * (cfg["seq_len"] - 1) * cfg["model_batch_size"]
        )  # clip to exact multiple of seq_len minus BOS

        self.buffer = self.init_buffer(self.model)

        self.first_fill = True  # whether first call to refresh has occurred
        self.pointer = 0

        self.res_layer = cfg["resid_layer"]

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

        for _ in tqdm.trange(0, num_batches, desc="Filling buffer..."):
            batch = next(self.dataloader_iter)

            _, cache = self.model.run_with_cache(
                batch["input_ids"],  # look at this?
                names_filter=lambda x: x.endswith("resid_post"),
                stop_at_layer=self.res_layer + 1,
            )
            cache: transformer_lens.ActivationCache

            acts = einops.rearrange(
                cache[get_act_name("resid_post", self.res_layer)][:, 1:],
                "batch seq_len d_model -> (batch seq_len) d_model",
            )

            self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
            self.pointer += acts.shape[0]

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

        if self.first_fill:
            self.scaling = (
                np.sqrt(self.model.cfg.d_model)
                / (self.buffer.norm(dim=-1).mean() + 1e-5).item()
            )

        self.first_fill = False

    def next(self) -> torch.Tensor:
        end: int = self.pointer + self.cfg["batch_size"]
        activations = self.buffer[self.pointer : end]
        self.pointer = end

        if self.pointer > self.buffer_size // 2:
            self.refresh()

        return activations * self.scaling
