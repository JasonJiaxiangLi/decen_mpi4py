# decen_mpi4py
A decentralized optimization framwork using mpi4py and pytorch for optimizer design

Building upon https://github.com/gmancino/DEEPSTORM

try

```
mpirun -np 8 python run.py --data='mnist' --model='lenet' --updates=10001 --report=100 --comm_pattern='random' --init_batch=1 --mini_batch=32 --step_type='constant' --k0=3 --beta=0.0228 --lr=10.0 --num_trial=1 --algorithm='dsgt' --device='cuda'
```
```
mpirun -np 8 python run.py --data='a9a' --model='mlp' --updates=10001 --report=100 --comm_pattern='random' --init_batch=1 --mini_batch=32 --step_type='constant' --k0=3 --beta=0.0228 --lr=10.0 --num_trial=1 --algorithm='dsgt'
```
```
mpirun -np 8 python run.py --data='cifar' --model='resnet' --updates=10001 --report=100 --comm_pattern='random' --init_batch=1 --mini_batch=32 --step_type='constant' --k0=3 --beta=0.0228 --lr=10.0 --num_trial=1 --algorithm='dsgt'
```
```
mpirun -np 8 python run.py --data='cifar' --model='resnet' --updates=10001 --report=100 --comm_pattern='random' --init_batch=1 --mini_batch=32 --step_type='constant' --k0=3 --beta=0.0228 --lr=10.0 --num_trial=1 --algorithm='dsgt' --device='cuda'
```