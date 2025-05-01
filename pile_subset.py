from datasets import load_dataset, Dataset, DatasetDict
from functools import partial

# pile = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")

# subset = list(pile.take(10000))

# ds = Dataset.from_list(subset, features=pile.features)

# ds.save_to_disk("datasets/pile-subset")

ds = Dataset.load_from_disk("datasets/pile-subset")

ds
