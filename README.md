# Model crosscoding

## Environment installation

I recommend using [`uv`](https://docs.astral.sh/uv/). It's easy to [install](https://docs.astral.sh/uv/getting-started/installation/). 

Then: `uv venv --python 3.11` to create a python 3.11 virtual environment.

`source .venv/bin/activate; uv pip install -r requirements.txt` will activate and install the environment.

You can also just use `python -m venv` and `pip`.

## Relevant papers

[Universal Sparse Autoencoders](https://arxiv.org/pdf/2502.03714)

[Scaling and evaluating sparse autoencoders](https://arxiv.org/pdf/2406.04093) (Good for implementation tips)

[k-Sparse Autoencoders](https://arxiv.org/pdf/1312.5663) (details on TopK autoencoders, which we should implement rather than ReLU autoencoders)

## Relevant codebases

[delphi](https://github.com/EleutherAI/delphi) (for doing feature explanations)

[Neel Nanda's crosscoders](https://github.com/neelnanda-io/Crosscoders) (implementation inspiration)