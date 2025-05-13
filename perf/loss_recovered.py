import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformer_lens import HookedTransformer, loading_from_pretrained
import transformer_lens.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from crosscoders.models import CrossCoder, BetterCrossCoder, SAE
import torch.nn.functional as F
from functools import partial


def normal_loss(model_name: str, dataset_name: str, dataset_split: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has padding tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    def tokenize(examples):
        tok = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        tok["labels"] = tok["input_ids"].clone()
        return tok

    def get_labels(examples):
        return {"labels": [x.clone() for x in examples["input_ids"]]}

    test_dataset = load_dataset(
        dataset_name, split=dataset_split, cache_dir="/oscar/scratch/pcurtin1/datasets"
    )

    # Tokenize with labels
    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
        remove_columns=test_dataset.features,
    )
    test_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=8)  # pyright: ignore

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loss = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Measuring normal loss..."):
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)
            loss += output.loss.to("cpu").item()

    return loss / len(test_dataloader)


def ablated_loss(
    model_name: str, model_layer: int, dataset_name: str, dataset_split: str
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has padding tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained_no_processing(
        model_name, tokenizer=tokenizer, dtype="bfloat16"
    )

    def tokenize(examples):
        tok = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        tok["labels"] = tok["input_ids"].clone()
        return tok

    test_dataset = load_dataset(
        dataset_name, split=dataset_split, cache_dir="/oscar/scratch/pcurtin1/datasets"
    )

    # Tokenize with labels
    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
        remove_columns=test_dataset.features,
    )
    test_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=8)  # pyright: ignore

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loss = 0

    # Function to replace residual values with ablated ones
    def ablate_residual(corrupted_residual_component, hook) -> torch.Tensor:
        # corrupted_residual_component's shape is "batch pos d_model"
        corrupted_residual_component[:, :, :] = corrupted_residual_component.mean(
            dim=-1, keepdim=True
        )

    SAE_influenced_layer = model_layer

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Measuring ablated loss..."):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Run the model with ablated loss
            patched_logits = model.run_with_hooks(
                batch["input_ids"],
                fwd_hooks=[
                    (
                        utils.get_act_name("resid_post", SAE_influenced_layer),
                        ablate_residual,
                    )
                ],  # May not be correct layer, off by one error
                return_type="logits",
            )

            ## Manually compute the loss, since we can't use the transformers API
            # Goal is to predict the next token, so we shift targets left by one
            # labels: [batch, seq_len]
            target_tokens = batch["labels"][:, 1:]
            # logits: [batch, seq_len, vocab_size]
            predicted_logits = patched_logits[:, :-1, :]  # Align with shifted targets

            # Compute cross-entropy loss
            cross_entropy_loss = F.cross_entropy(
                predicted_logits.reshape(-1, predicted_logits.size(-1)),
                target_tokens.reshape(-1),
                reduction="mean",
            )

            loss += cross_entropy_loss.to("cpu").item()

    return loss / len(test_dataloader)


def crosscoder_loss(
    model_a_name: str,
    model_a_layer: int,
    model_a_dim: int,
    model_b_name: str,
    model_b_layer: int,
    sae_model_name: str,
    sae_model_path: str,
    dataset_name: str,
    dataset_split: str,
    right: bool = False,
    pure: bool = False,
    better: bool = False,
):
    # Load in the models and tokenizer - the crosscoder requires HookedTransformers
    # NOTE: This assumes that the models share a tokenizer - big assumption

    tokenizer = AutoTokenizer.from_pretrained(
        model_a_name
    )  # Using model_a's name arbitrarily

    model_a = HookedTransformer.from_pretrained_no_processing(
        model_a_name, tokenizer=tokenizer, dtype="bfloat16"
    )

    model_b = HookedTransformer.from_pretrained_no_processing(
        model_b_name, tokenizer=tokenizer, dtype="bfloat16"
    )

    # Ensure tokenizer has padding tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # model_a.resize_token_embeddings(len(tokenizer))
    # model_b.resize_token_embeddings(len(tokenizer))

    def tokenize(examples):
        tok = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        tok["labels"] = tok["input_ids"].clone()
        return tok

    test_dataset = load_dataset(
        dataset_name, split=dataset_split, cache_dir="/oscar/scratch/pcurtin1/datasets"
    )

    # Tokenize with labels
    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
        remove_columns=test_dataset.features,
    )
    test_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=8)  # pyright: ignore

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a = model_a.to(device)
    model_a.eval()
    model_b = model_b.to(device)
    model_b.eval()

    loss = 0

    # Load the SAE
    cross_model = BetterCrossCoder if better else CrossCoder
    crosscoder = cross_model.load(
        sae_model_name,
        model_dim_a=model_a.cfg.d_model,
        model_dim_b=model_b.cfg.d_model,
        path=sae_model_path,
    )
    crosscoder.to(device)

    # Function to give to the hook
    def patch_residual_with_SAE(
        corrupted_residual_component, hook, crosscoder, model_cache, cache_layer
    ) -> torch.Tensor:
        # corrupted_residual_component's shape is "batch pos d_model"
        # Retrieve the residuals we got from running model_a at the same location
        model_residuals = model_cache[
            cache_layer
        ]  # same shape as corrupted_residual_component

        # Combine the current residual with the residual from model_a_cache to be how crosscoder expects it

        if pure:
            corrupted_copy = torch.zeros_like(corrupted_residual_component)
        else:
            corrupted_copy = corrupted_residual_component

        if right:
            x = torch.cat((model_residuals, corrupted_copy), dim=-1)
        else:
            x = torch.cat((corrupted_copy, model_residuals), dim=-1)

        if better:
            if pure:
                x = model_residuals
            else:
                x = corrupted_residual_component

        # Run the crosscoder and do some wrangling of shapes so we separate the reconstructed residuals by model

        # right, pure -> outcome
        # 0, 0 -> 0
        # 1, 0 -> 1
        # 0, 1 -> 1
        # 1, 1 -> 0

        m_in, m_out = right ^ pure, right
        reconstructed = crosscoder(x, m_in, m_out) if better else crosscoder(x)
        # Shape is 8 (batch_size) x 512 (token_len) x 1792 (neurons)

        # separated_reconstructed_residuals = reconstructed.reshape(
        #     reconstructed.shape[0],
        #     reconstructed.shape[1],
        #     2,
        #     reconstructed.shape[2] // 2,
        # )
        # model_b_reconstructed_residuals = separated_reconstructed_residuals[:, :, 1, :]
        if better:
            model_reconstructed_residuals = reconstructed
        else:
            if right:
                model_reconstructed_residuals = reconstructed[:, :, model_a_dim:]
            else:
                model_reconstructed_residuals = reconstructed[:, :, :model_a_dim]

        # Replace the residual with the results we get from the crosscoder
        corrupted_residual_component[:, :, :] = (
            model_reconstructed_residuals  # model_b_reconstructed_residuals
        )

    # SAE_influenced_layer = 18

    cache_model = model_a if right else model_b
    cache_model_layer = model_a_layer if right else model_b_layer
    hook_model = model_b if right else model_a
    hook_model_layer = model_b_layer if right else model_a_layer

    # print("hook model:", hook_model)
    # print("cache model:", cache_model)

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Measuring SAE loss..."):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Run the first model normally, and get the cache
            _, model_cache = cache_model.run_with_cache(
                batch["input_ids"],
                stop_at_layer=cache_model_layer + 1,
            )

            # Run the second model with a hook, and give it the first model's activations
            hook_fn = partial(
                patch_residual_with_SAE,
                crosscoder=crosscoder,
                model_cache=model_cache,
                cache_layer=utils.get_act_name("resid_post", cache_model_layer),
            )
            patched_logits = hook_model.run_with_hooks(
                batch["input_ids"],
                fwd_hooks=[
                    (utils.get_act_name("resid_post", hook_model_layer), hook_fn)
                ],
                return_type="logits",
            )

            ## Manually compute the loss, since we can't use the transformers API
            # Goal is to predict the next token, so we shift targets left by one
            # labels: [batch, seq_len]
            target_tokens = batch["labels"][:, 1:]
            # logits: [batch, seq_len, vocab_size]
            predicted_logits = patched_logits[:, :-1, :]  # Align with shifted targets

            # Compute cross-entropy loss
            cross_entropy_loss = F.cross_entropy(
                predicted_logits.reshape(-1, predicted_logits.size(-1)),
                target_tokens.reshape(-1),
                reduction="mean",
            )

            loss += cross_entropy_loss.to("cpu").item()

    return loss / len(test_dataloader)


def sae_loss(
    model_name: str,
    sae_model_name: str,
    sae_model_path: str,
    dataset_name: str,
    dataset_split: str,
):
    ## LOADING AND TOKENIZING
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has padding tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HookedTransformer.from_pretrained_no_processing(
        model_name, tokenizer=tokenizer, dtype="bfloat16"
    )

    def tokenize(examples):
        tok = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        tok["labels"] = tok["input_ids"].clone()
        return tok

    test_dataset = load_dataset(
        dataset_name, split=dataset_split, cache_dir="/oscar/scratch/pcurtin1/datasets"
    )

    # Tokenize with labels
    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
        remove_columns=test_dataset.features,
    )
    test_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=8)  # pyright: ignore

    # Load the SAE
    model_dim = loading_from_pretrained.get_pretrained_model_config(model_name).d_model
    sae = SAE.load(sae_model_name, model_dim, path=sae_model_path)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae.to(device)
    model = model.to(device)
    model.eval()

    loss = 0

    ## CREATE HOOK AND TRAIN MODEL
    # Function to replace residual values with ablated ones
    def patch_residual_with_SAE(
        corrupted_residual_component, hook, SAE
    ) -> torch.Tensor:
        # corrupted_residual_component's shape is "batch pos d_model"
        # Replace the residual with the result after running it through an SAE
        corrupted_residual_component[:, :, :] = SAE(corrupted_residual_component)

    SAE_influenced_layer = 18

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Measuring SAE loss..."):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Run the model with ablated loss
            hook_fn = partial(patch_residual_with_SAE, SAE=sae)
            patched_logits = model.run_with_hooks(
                batch["input_ids"],
                fwd_hooks=[
                    (utils.get_act_name("resid_post", SAE_influenced_layer), hook_fn)
                ],  # May not be correct layer, off by one error
                return_type="logits",
            )

            ## Manually compute the loss, since we can't use the transformers API
            # Goal is to predict the next token, so we shift targets left by one
            # labels: [batch, seq_len]
            target_tokens = batch["labels"][:, 1:]
            # logits: [batch, seq_len, vocab_size]
            predicted_logits = patched_logits[:, :-1, :]  # Align with shifted targets

            # Compute cross-entropy loss
            cross_entropy_loss = F.cross_entropy(
                predicted_logits.reshape(-1, predicted_logits.size(-1)),
                target_tokens.reshape(-1),
                reduction="mean",
            )

            loss += cross_entropy_loss.to("cpu").item()

    return loss / len(test_dataloader)


if __name__ == "__main__":
    # Recover loss when running like normal
    loss_n = normal_loss(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "EleutherAI/fineweb-edu-dedup-10b",
        "train[:2500]",
    )

    # Recover loss when ablating residuals to be the mean
    loss_a = ablated_loss(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "EleutherAI/fineweb-edu-dedup-10b",
        "train[:2500]",
    )

    # Recover loss when using the SAE to replace the residuals
    loss_s = crosscoder_loss(
        model_a_name="Qwen/Qwen2.5-0.5B",
        model_b_name="Qwen/Qwen2.5-0.5B-Instruct",
        sae_model_name=9,
        sae_model_path="../models/some_model/version_1",
        dataset_name="EleutherAI/fineweb-edu-dedup-10b",
        dataset_split="train[:2500]",
    )

    # sae_loss = sae_loss(
    #     model_name="Qwen/Qwen2.5-0.5B-Instruct",
    #     sae_model_name="9",
    #     sae_model_path="../models/qwen_instruct_sae/version_0",
    #     dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    #     dataset_split="train[:2500]",
    # )

    print(f"Loss reconstructed: {(loss_a - loss_s) / (loss_a - loss_n):.2f}")
