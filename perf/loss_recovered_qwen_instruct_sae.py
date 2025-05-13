from loss_recovered import normal_loss, ablated_loss, sae_loss

# Recover loss when running like normal
loss_n = normal_loss(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
)

# Recover loss when ablating residuals to be the mean
loss_a = ablated_loss(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    model_layer=18,
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
)

# Recover loss when using the SAE to replace the residuals
loss_s = sae_loss(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    sae_model_name=9,
    sae_model_path="../models/qwen_instruct_sae/version_0",
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
)

print(f"Loss reconstructed: {(loss_a - loss_s) / (loss_a - loss_n):.2f}")
