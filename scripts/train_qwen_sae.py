import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from crosscoders.training import SAETrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

cfg = {
    "seed": 51,
    "batch_size": 2048,
    "buffer_mult": 512,
    "lr": 1e-3,
    "num_tokens": int(1e8),
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "dict_size": 2**15,
    "topk": 64,
    "seq_len": 512,
    "device": "cuda:0",
    "model_batch_size": 8,
    "log_every": 100,
    "save_every": 5000,
    "dec_init_norm": 0.05,
    "save_dir": "models/qwen_instruct_sae",
    "save_version": 0,
    "resid_layer": 18,
}


model_str = "Qwen/Qwen2.5-0.5B-Instruct"

########

pile = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_str)

tokenizer.pad_token = tokenizer.eos_token


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=cfg["seq_len"],
        truncation=True,
        return_tensors="pt",
    )


pile = pile.map(
    lambda x: tokenize(x, tokenizer),
    batched=True,
    remove_columns=["meta", "attention_mask"],
)

dl = DataLoader(dataset=pile, batch_size=cfg["model_batch_size"])

model = HookedTransformer.from_pretrained(
    model_str, tokenizer=tokenizer, dtype="bfloat16"
)
for param in model.parameters():
    param.requires_grad = False


t = SAETrainer(model, dl, use_wandb=True, cfg=cfg)
t.train()
