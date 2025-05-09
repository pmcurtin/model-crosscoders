import os

os.environ["HF_HOME"] = "/oscar/scratch/pcurtin1"

from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from typing import Callable
import torch
from crosscoders.models import SAE
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.utils import assert_type
from pathlib import Path
from delphi.__main__ import (
    non_redundant_hookpoints,
    create_neighbours,
    process_cache,
    populate_cache,
    log_results,
)

import asyncio
import time
from delphi.log.result_analysis import get_metrics, load_data


# model_path = "../models/qwen_instruct_sae/version_0/"
# name = "9"
# layer = 18

# model_str = "Qwen/Qwen2.5-0.5B-Instruct"
# model_dim = 896

# run_name = "sae"

#############


def delphi_autointerp(
    model_path: str,
    checkpoint: str,
    model_str: str,
    model_dim: int,
    layer: int,
    run_name: str,
):
    def override_load_hooks_sparse_coders(
        model: AutoModel, cfg: RunConfig, compile: bool = False
    ) -> tuple[dict[str, Callable], bool]:
        coder = SAE.load(checkpoint, model_dim=model_dim, path=model_path).to("cuda")

        transcoder = False

        assert len(cfg.hookpoints) == 1, "only support one hook location"

        d = {cfg.hookpoints[0]: coder.encode}

        return d, transcoder

    def load_artifacts(run_cfg: RunConfig):
        if run_cfg.load_in_8bit:
            dtype = torch.float16
        elif torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = "auto"

        model = AutoModel.from_pretrained(
            run_cfg.model,
            device_map={"": "cuda"},
            quantization_config=(
                BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
                if run_cfg.load_in_8bit
                else None
            ),
            torch_dtype=dtype,
            token=run_cfg.hf_token,
        )

        hookpoint_to_sparse_encode, transcode = override_load_hooks_sparse_coders(
            model,
            run_cfg,
            compile=True,
        )

        return (
            list(hookpoint_to_sparse_encode.keys()),
            hookpoint_to_sparse_encode,
            model,
            transcode,
        )

    async def run(
        run_cfg: RunConfig,
    ):
        base_path = Path.cwd() / "results"
        if run_cfg.name:
            base_path = base_path / run_cfg.name

        base_path.mkdir(parents=True, exist_ok=True)

        run_cfg.save_json(base_path / "run_config.json", indent=4)

        latents_path = base_path / "latents"
        explanations_path = base_path / "explanations"
        scores_path = base_path / "scores"
        neighbours_path = base_path / "neighbours"
        visualize_path = base_path / "visualize"

        latent_range = (
            torch.arange(run_cfg.max_latents) if run_cfg.max_latents else None
        )

        hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(
            run_cfg
        )
        tokenizer = AutoTokenizer.from_pretrained(run_cfg.model, token=run_cfg.hf_token)

        nrh = assert_type(
            dict,
            non_redundant_hookpoints(
                hookpoint_to_sparse_encode, latents_path, "cache" in run_cfg.overwrite
            ),
        )
        if nrh:
            populate_cache(
                run_cfg,
                model,
                nrh,
                latents_path,
                tokenizer,
                transcode,
            )

        del model, hookpoint_to_sparse_encode
        if run_cfg.constructor_cfg.non_activating_source == "neighbours":
            nrh = assert_type(
                list,
                non_redundant_hookpoints(
                    hookpoints, neighbours_path, "neighbours" in run_cfg.overwrite
                ),
            )
            if nrh:
                create_neighbours(
                    run_cfg,
                    latents_path,
                    neighbours_path,
                    nrh,
                )
        else:
            print("Skipping neighbour creation")

        nrh = assert_type(
            list,
            non_redundant_hookpoints(
                hookpoints, scores_path, "scores" in run_cfg.overwrite
            ),
        )
        if nrh:
            await process_cache(
                run_cfg,
                latents_path,
                explanations_path,
                scores_path,
                nrh,
                tokenizer,
                latent_range,
            )

        if run_cfg.verbose:
            log_results(scores_path, visualize_path, run_cfg.hookpoints)

    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:50000]",
        dataset_column="text",
        batch_size=8,
        cache_ctx_len=256,
        n_splits=5,
        n_tokens=5_000_000,
    )
    sampler_cfg = SamplerConfig(
        train_type="quantiles",
        test_type="quantiles",
        n_examples_train=30,
        n_examples_test=20,
        n_quantiles=10,
    )
    constructor_cfg = ConstructorConfig(
        min_examples=50,
        example_ctx_len=32,
        n_non_activating=20,
        non_activating_source="random",
        faiss_embedding_cache_enabled=True,
        faiss_embedding_cache_dir=".embedding_cache",
    )
    run_cfg = RunConfig(
        name=run_name,
        overwrite=["cache", "scores"],
        model=model_str,
        sparse_model="EleutherAI/sae-pythia-160m-32k",
        hookpoints=[f"layers.{layer}"],
        explainer_model="Qwen/Qwen2.5-7B-Instruct",  # "meta-llama/Llama-3.2-3B-Instruct",
        # explainer_model_max_len=4096,
        # max_latents=100,
        seed=21,
        num_gpus=torch.cuda.device_count(),
        filter_bos=True,
        verbose=True,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        cache_cfg=cache_cfg,
    )

    start_time = time.time()
    asyncio.run(run(run_cfg))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    scores_path = Path.cwd() / "results" / run_cfg.name / "scores"

    latent_df, _ = load_data(scores_path, run_cfg.hookpoints)
    processed_df = get_metrics(latent_df)
    # Performs better than random guessing
    for score_type, df in processed_df.groupby("score_type"):
        accuracy = df["accuracy"].mean()
        assert accuracy > 0.55, f"Score type {score_type} has an accuracy of {accuracy}"
