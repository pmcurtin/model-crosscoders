# Model crosscoding

## Environment installation

We recommend using [`uv`](https://docs.astral.sh/uv/). It's easy to [install](https://docs.astral.sh/uv/getting-started/installation/). 

Then: `uv venv --python 3.11` to create a python 3.11 virtual environment.

`source .venv/bin/activate; uv pip install -r requirements.txt` will activate and install the environment.

You can also just use `python -m venv` and `pip`.

### Setting up `crosscoders`

After installing the environment, you should install our editable `crosscoders` package. This is achieved by running: `uv pip install -e .` or `pip install -e .` in the activated environment.

## Important scripts

The `scripts/` directory contains training scripts for the various crosscoders we trained. 

The `perf/` directory contains scripts for evaluating the loss recovered by various crosscoder models. 

The `interp/` directory contains scripts for evaluating crosscoder interpretability using delphi, and creating the relative norm strength plots. 

## Relevant codebases

### Delphi

In the `interp/` directory, the `run_delphi_*` scripts require [delphi](https://github.com/EleutherAI/delphi) to be installed in the environment.

### Reference Crosscoder implementation

We loosely followed the training architecture established in [Neel Nanda's crosscoders](https://github.com/neelnanda-io/Crosscoders) repositor, but wrote our own crosscoder classes (in `crosscoders/models.py`) and trainers to accomodate crosscoding between two models.