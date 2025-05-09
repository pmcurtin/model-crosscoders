from loss_recovered import normal_loss, ablated_loss, sae_loss

assert False, "this is not done do not use this"

# Recover loss when running like normal
loss_n = normal_loss(
    "Qwen/Qwen2.5-0.5B",
    "EleutherAI/fineweb-edu-dedup-10b",
    "train[:2500]",
)

# Recover loss when ablating residuals to be the mean
loss_a = ablated_loss(
    "Qwen/Qwen2.5-0.5B",
    "EleutherAI/fineweb-edu-dedup-10b",
    "train[:2500]",
)

# Recover loss when using the SAE to replace the residuals
loss_s = sae_loss(
    model_a_name="Qwen/Qwen2.5-0.5B",
    model_b_name="Qwen/Qwen2.5-0.5B-Instruct",
    sae_model_name=9,
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
)

print(f"Loss reconstructed: {(loss_a - loss_s) / (loss_a - loss_n):.2f}")
