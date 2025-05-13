from torch import mode
from loss_recovered import crosscoder_loss, normal_loss, ablated_loss

# Recover loss when running like normal
loss_n = normal_loss(
    model_name="Qwen/Qwen2.5-0.5B",
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
)

# Recover loss when ablating residuals to be the mean
loss_a = ablated_loss(
    model_name="Qwen/Qwen2.5-0.5B",
    model_layer=18,
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
)

# Recover loss when using the SAE to replace the residuals
loss_s = crosscoder_loss(
    model_a_name="Qwen/Qwen2.5-0.5B",
    model_a_layer=18,
    model_b_name="Qwen/Qwen2.5-0.5B-Instruct",
    model_b_layer=18,
    model_a_dim=896,
    sae_model_name=9,
    sae_model_path="../models/some_model/version_1",  # "../models/qwen_crosscoder_better/version_3",
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
    right=False,
    pure=False,
    better=False,
)

print(f"Loss reconstructed: {(loss_a - loss_s) / (loss_a - loss_n):.2f}")
