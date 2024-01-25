# decen_mpi4py
A decentralized optimization framwork using mpi4py and pytorch for optimizer design

Building upon https://github.com/gmancino/DEEPSTORM

try

```
mpirun -np 8 python run.py --data='mnist' --updates=10001 --report=100 --comm_pattern='random' --init_batch=1 --mini_batch=32 --step_type='constant' --k0=3 --beta=0.0228 --lr=10.0 --num_trial=1 --algorithm='dsgt'
```