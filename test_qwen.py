from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt
import torch

model_str = "Qwen/Qwen2.5-0.5B"


model = HookedTransformer.from_pretrained_no_processing(model_str, dtype=torch.bfloat16)

print(test_prompt("How old are", "you", model, prepend_bos=False))
