from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt
import torch

# model_str = "EleutherAI/pythia-160m" # 11 layers
# model_str = "EleutherAI/pythia-410m-v0" # 23 layers

model_str = 'Qwen/Qwen2.5-0.5B'

model = HookedTransformer.from_pretrained_no_processing(model_str, dtype=torch.bfloat16)

print(test_prompt("How old are", "you", model, prepend_bos=True))


# tok = AutoTokenizer.from_pretrained(model_str)
# print(tok.vocab_size)
# print(tok.encode(tok.eos_token))
# print(tok.bos_token)
# print(tok.encode("How are you doing today"))
# print(tok.convert_tokens_to_ids(tok.eos_token))
# print(tok.convert_ids_to_tokens(2347))
# tokens = model.to_tokens("how are we doing tonight fellas", prepend_bos=False)
# logits, cache = model.run_with_cache(tokens)
# print(list(cache.keys()))
