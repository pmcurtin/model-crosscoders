import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from config import default_cfg
from training import CrossCoderTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

cfg = default_cfg

pile = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")

# dl = DataLoader(dataset=pile, batch_size=cfg["model_batch_size"])

model_a_str = "gpt2-medium"  # "Qwen/Qwen2.5-0.5B"
model_b_str = "microsoft/DialoGPT-medium"  # "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer_a = AutoTokenizer.from_pretrained(model_a_str)
tokenizer_b = AutoTokenizer.from_pretrained(model_b_str)

tokenizer_a.pad_token = tokenizer_a.eos_token
tokenizer_b.pad_token = tokenizer_b.eos_token


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=cfg["seq_len"],
        truncation=True,
        return_tensors="pt",
    )


pile = (
    pile.map(
        lambda x: tokenize(x, tokenizer_a),
        batched=True,
        remove_columns=["meta", "attention_mask"],
    )
    .rename_column("input_ids", "input_ids_a")
    .map(
        lambda x: tokenize(x, tokenizer_b),
        batched=True,
        remove_columns=["text", "attention_mask"],
    )
    .rename_column("input_ids", "input_ids_b")
)

dl = DataLoader(dataset=pile, batch_size=cfg["model_batch_size"])
# dl_b = DataLoader(dataset=dataset_b, batch_size=cfg["model_batch_size"])

model_a = HookedTransformer.from_pretrained(
    model_a_str, tokenizer=tokenizer_a, dtype="bfloat16"
)
for param in model_a.parameters():
    param.requires_grad = False
model_b = HookedTransformer.from_pretrained(
    model_a_str,
    tokenizer=tokenizer_b,
    dtype="bfloat16",
    hf_model=AutoModelForCausalLM.from_pretrained(model_b_str),
)
for param in model_b.parameters():
    param.requires_grad = False

t = CrossCoderTrainer(model_a, model_b, dl, use_wandb=True, use_better=False)
t.train()
