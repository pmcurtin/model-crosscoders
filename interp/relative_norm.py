import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Import and grab decoder weights
CROSSCODER_RUN_NAME = "flowing-durian-75"
MODEL_PATH = "../models/some_model/version_1/9.pt"

MODEL_A_NAME = "Qwen2.5-0.5B"
MODEL_B_NAME = "Qwen2.5-0.5B-Instruct"

better = False

model_a_dim = 896

model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

if better:
    decoder_a_norms = (model["decoder_a"].cpu().float()).norm(
        dim=-1
    )  # decoder_weights[:, :768].norm(dim=-1)
    decoder_b_norms = (model["decoder_b"].cpu().float()).norm(
        dim=-1
    )  # decoder_weights[:, 768:].norm(dim=-1)

else:
    decoder_weights = model["decoder"].cpu().float()

    decoder_a_norms = decoder_weights[:, :model_a_dim].norm(dim=-1)

    decoder_b_norms = decoder_weights[:, model_a_dim:].norm(dim=-1)

    # print(model["decoder"].shape)
    # Calculate norm based on model activations and calculate relative strength
    # decoder_weights = model["decoder"].cpu().float()  # Upcast bfloat to float for numpy
    # Decoder_weights is 32768 (dictionary size) x 1792 (neurons)
    # reshape to 32768 x 2 x 896 ([:, 0] contains decoder A, and [:, 1] contains decoder B)

# decoder_reshaped = decoder_weights.reshape(decoder_weights.shape[0], 2, -1)

# norms = decoder_reshaped.norm(dim=-1)
# relative_norms = norms[:, 1] / norms.sum(dim=-1)
relative_norms = decoder_b_norms / (decoder_a_norms + decoder_b_norms)

pd.DataFrame(
    {
        "latent_idx": range(len(relative_norms)),
        "model_a_norm": decoder_a_norms,
        "model_b_norm": decoder_b_norms,
        "relative_norm": relative_norms,
    }
).to_csv("norm_strengths.csv", index=False)

# Make histogram to visualize the distribution of the normalized L2 norms
plt.figure(figsize=(10, 6))
sns.histplot(data=relative_norms, bins=100)

# Add titles and labels
plt.title(
    f"Crosscoder {CROSSCODER_RUN_NAME} Relative Decoder Norms for {MODEL_A_NAME} and {MODEL_B_NAME}"
)
plt.xlabel("Relative Decoder Norm Strength")
plt.ylabel("Number of Features")
plt.xlim((0, 1))

# Save the plot
if not os.path.exists("../figures"):
    os.makedirs("../figures")
plt.savefig(f"../figures/{CROSSCODER_RUN_NAME} Relative L2 Norm Graph.png")
