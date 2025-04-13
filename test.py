import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from config import default_cfg
from training import CrossCoderTrainer
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

cfg = default_cfg

pile = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")

# dl = DataLoader(dataset=pile, batch_size=cfg["model_batch_size"])

model_a_str = "Qwen/Qwen2.5-0.5B"
model_b_str = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer_a = AutoTokenizer.from_pretrained(model_a_str)
tokenizer_b = AutoTokenizer.from_pretrained(model_b_str)


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=cfg["seq_len"],
        truncation=True,
        return_tensors="pt",
    )


dataset_a = pile.map(
    lambda x: tokenize(x, tokenizer_a), batched=True, remove_columns=["text", "meta"]
)

dataset_b = pile.map(
    lambda x: tokenize(x, tokenizer_b), batched=True, remove_columns=["text", "meta"]
)

dl_a = DataLoader(dataset=dataset_a, batch_size=cfg["model_batch_size"])
dl_b = DataLoader(dataset=dataset_b, batch_size=cfg["model_batch_size"])

model_a = HookedTransformer.from_pretrained(model_a_str, tokenizer=tokenizer_a)
for param in model_a.parameters():
    param.requires_grad = False
model_b = HookedTransformer.from_pretrained(model_b_str, tokenizer=tokenizer_b)
for param in model_b.parameters():
    param.requires_grad = False

t = CrossCoderTrainer(model_a, model_b, dl_a, dl_b)
t.train()
