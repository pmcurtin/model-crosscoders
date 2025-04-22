import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import and grab decoder weights
MODEL_NAME = "flowing-durian-75"
model = torch.load(
    f"./models/some_model/version_1/9.pt", map_location=torch.device("cpu")
)

# Calculate norm based on model activations and calculate relative strength
decoder_weights = model["decoder"].cpu().float()  # Upcast bfloat to float for numpy

# Decoder_weights is 32768 (dictionary size) x 1792 (neurons)

# reshape to 32768 x 2 x 896 ([:, 0] contains decoder A, and [:, 1] contains decoder B)
decoder_reshaped = decoder_weights.reshape(decoder_weights.shape[0], 2, -1)

norms = decoder_reshaped.norm(dim=-1)
relative_norms = norms[:, 1] / norms.sum(dim=-1)

# norm = np.linalg.norm(decoder_weights, axis=1) # Each row represents one feature? So we compute the L2 norm
# normalized_norms = norm / np.max(norm)

# Make histogram to visualize the distribution of the normalized L2 norms
plt.figure(figsize=(10, 6))
sns.histplot(data=relative_norms, bins=100)

# Add titles and labels
plt.title(
    f"Crosscoder {MODEL_NAME} Decoder Weights for Base and Finetuned Qwen 2.5-0.5B"
)
plt.xlabel("Relative Decoder Norm Strength")
plt.ylabel("Number of Features")

# Save the plot
if not os.path.exists("./figures"):
    os.makedirs("./figures")
plt.savefig(f"./figures/{MODEL_NAME} Relative L2 Norm Graph.png")
