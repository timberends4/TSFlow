# Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting

Marcel Kollovieh, Marten Lienen, David Lüdke, Leo Schwinn, Stephan Günnemann


## Installation

If you want to run our code, start by setting up the python environment.
We use [pixi](https://pixi.sh/) to easily set up reproducible environments based on conda packages.
Install it with `curl -fsSL https://pixi.sh/install.sh | bash` and then run

```sh
# Clone the repository
git clone https://github.com/marcelkollovieh/TSFlow.git

# Change into the repository
cd TSFlow

# Install and activate the environment
pixi shell
```

## Training

Start a training by running `train.py` with the your settings, for example
```sh
python bin/train_model.py -c configs_local/train_conditional.yaml
```

The results will be logged in `./logs`

## Citation

If you build upon this work, please cite our paper as follows.

```bibtex
@article{kollovieh2024flow,
  title = {Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting},
  author = {Kollovieh, Marcel and Lienen, Marten and L{\"u}dke, David and Schwinn, Leo and G{\"u}nnemann, Stephan},
  journal = {The Thirteenth International Conference on Learning Representations},
  shortjournal = {ICLR},
  year = {2025},
}
```