from loss_recovered import crosscoder_loss, normal_loss, ablated_loss

# Recover loss when running like normal
loss_n = normal_loss(
    model_name="EleutherAI/pythia-410m-v0",
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
)

# Recover loss when ablating residuals to be the mean
loss_a = ablated_loss(
    model_name="EleutherAI/pythia-410m-v0",
    model_layer=23,
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
)

# Recover loss when using the SAE to replace the residuals
loss_s = crosscoder_loss(
    model_a_name="EleutherAI/pythia-160m",
    model_a_layer=11,
    model_b_name="EleutherAI/pythia-410m-v0",
    model_b_layer=23,
    model_a_dim=768,
    sae_model_name=9,
    sae_model_path="../models/pythia_crosscoder/version_0",  # "../models/pythia_crosscoder_better/version_1",  #
    dataset_name="EleutherAI/fineweb-edu-dedup-10b",
    dataset_split="train[:2500]",
    right=True,
    pure=False,
    better=False,
)

print(f"Loss reconstructed: {(loss_a - loss_s) / (loss_a - loss_n):.2f}")
