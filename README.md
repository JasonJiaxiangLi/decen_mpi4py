# decen_mpi4py
A decentralized optimization framework using mpi4py and pytorch for optimizer design

## Installation
This code is built upon https://github.com/gmancino/DEEPSTORM, you would need mpi4py and pytorch installed:

### Install PyTorch

Follow the instructions on the [PyTorch](https://pytorch.org/get-started/previous-versions/) webpage for installation.

### Install mpi4py

```
conda install -c anaconda gcc_linux-64
conda install -c conda-forge mpi4py openmpi
```

## Run code
try running in the following way

```
mpirun -np 8 python run.py --data='mnist' --model='lenet' --updates=10001 --report=100 --comm_pattern='random' --init_batch=1 --mini_batch=32 --step_type='constant' --k0=3 --beta=0.0228 --lr=10.0 --num_trial=1 --algorithm='dsgt' --device='cpu'
```

## About results and plots
The "results" folder contains all the experiment results for plotting, and the Jupyter notebooks beginning with "plots" . Note: we plot before we name all the algorithms, so the names are not very consistent.
