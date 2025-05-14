from datasets import load_dataset, Dataset, DatasetDict
from functools import partial

# Hacky way to load a subset of the Pile dataset, uncomment each line individually then run
# to create a subset of the dataset and save it to disk

# pile = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")

# subset = list(pile.take(10000))

# ds = Dataset.from_list(subset, features=pile.features)

# ds.save_to_disk("datasets/pile-subset")

ds = Dataset.load_from_disk("datasets/pile-subset")

ds
