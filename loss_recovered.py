import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        for batch in tqdm(test_dataloader, desc="Measuring loss..."):
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)
            loss += output.loss.to("cpu").item()

    return loss / len(test_dataloader)


if __name__ == "__main__":
    print(
        normal_loss(
            "Qwen/Qwen2.5-0.5B", "EleutherAI/fineweb-edu-dedup-10b", "train[:5000]"
        )
    )
