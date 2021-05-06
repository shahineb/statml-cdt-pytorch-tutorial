# Introduction to PyTorch

_StatML 2020/2021 CDT Introduction to PyTorch tutorial_


This repository contains 3 notebooks
- `01-introduction.ipynb`: Introduction to the fundamentals of PyTorch and automatic differentation
- `02-deep-kernel-learning-challenge-student.ipynb`: Hands-on implementation of deep kernel learning with PyTorch
- `03-dataloading-with-pytorch.ipynb`: Basics of dataloading pipelines with PyTorch



## Installation

For this tutorial, you are roughly going to need a Jupyter kernel with `torch`, `gpytorch`, `torchvision`, and usual scientific libraries installed. If your usual python environment already has all of this, you can go ahead and use it. Otherwise, follow the below instruction to set up an environment.


Code implemented with Python 3.8. Instructions written for 3.8.0 but can be adapted.


### Clone and go to repository
```bash
$ git clone https://github.com/shahineb/statml-cdt-pytorch-tutorial.git
$ cd statml-cdt-pytorch-tutorial
```

### Setting up environment

Create and activate a dedicated environment with your favorite virtual environment management tool.

__With virtualenv :__
```bash
$ virtualenv --python=python3.8 venv-pytorch-tutorial
$ source venv-pytorch-tutorial/bin/activate
$ (venv-pytorch-tutorial)
```


__With pyenv :__
```bash
$ pyenv virtualenv 3.8.0 venv-pytorch-tutorial
$ pyenv activate venv-pytorch-tutorial
$ (venv-pytorch-tutorial)
```


__With conda :__
```bash
$ conda create --name venv-pytorch-tutorial python=3.8
$ source activate venv-pytorch-tutorial
$ (venv-pytorch-tutorial)
```

### Install dependencies

```bash
$ (venv-pytorch-tutorial) pip install -r requirements.txt
```

### Create Jupyter Kernel

```bash
$ (venv-pytorch-tutorial) python -m ipykernel install --user --name pytorch-tutorial  --display-name "pytorch-tutorial"
```

You can now choose kernel named `pytorch-tutorial` from Jupyter.
