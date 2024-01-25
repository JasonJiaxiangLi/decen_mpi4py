# Import packages
from __future__ import print_function
import argparse
import os
import sys
import time
import torch
import numpy as np
import math
from mpi4py import MPI
from torchvision import datasets, transforms

# Import custom classes
from models.mlp import MLP
from models.lenet import LENET
from helpers.l1_regularizer import L1
from helpers.replace_weights import Opt
from helpers.custom_data_loader import BinaryDataset

###############################
# ADD NEW algorithm here
# also need to add in parse_args
from algorithms.dsgd import DSGD
from algorithms.dsgt import DSGT
from algorithms.dasagt import DASAGT
from algorithms.dnasa import DNASA
solver_dict = {"dsgd": DSGD, "dsgt": DSGT, "dasagt": DASAGT, "dnasa": DNASA}
###############################

# Set up MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

###############################
# read all the args
def parse_args():
    # Parse user input
    parser = argparse.ArgumentParser(description='Testing algorithms on problems from paper.')

    parser.add_argument('--algorithm', type=str, default='dsgt', choices=['dsgd', 'dsgt', 'dasagt', 'dnasa'], 
                        help='The algorithm you want to test.')
    parser.add_argument('--updates', type=int, default=5000, help='Total number of communication rounds.')
    parser.add_argument('--lr', type=float, default=1.0, help='Local learning rate.')
    parser.add_argument('--alpha_base', type=float, default=0.3, help='Moving average rate base')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta numerator coefficient.')
    parser.add_argument('--k0', type=int, default=5, help='Step-size term.')
    parser.add_argument('--l1', type=float, default=0.0, help='L-1 Regularizer.')
    parser.add_argument('--mini_batch', type=int, default=64, help='Mini-batch size.')
    parser.add_argument('--init_batch', type=int, default=1, help='Initial batch size.')
    parser.add_argument('--comm_pattern', type=str, default='ring', choices=['ring', 'random', 'complete', 'ladder'],
                        help='Communication pattern.')
    parser.add_argument('--comm_round', type=int, default=1, help='m')
    parser.add_argument('--data', type=str, default='a9a', choices=['a9a', 'mnist', 'miniboone', 'cifar'],
                        help='Dataset.')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'lenet', 'resnet'],
                        help='Neural network structure.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='The device.')
    parser.add_argument('--num_trial', type=int, default=1, help='Total number of trials.')
    parser.add_argument('--step_type', type=str, default='diminishing', choices=('constant', 'diminishing'),
                        help='Diminishing or constant step-size.')
    parser.add_argument('--report', type=int, default=100, help='How often to report criteria.')
    parser.add_argument('--init_seed_list', type=list, default=[], help='The random seed for initializing all parameters.')

    # Create callable argument
    args = parser.parse_args()
    
    if rank == 1:
        print(args)
    return args

args = parse_args()
if not args.init_seed_list:
    args.init_seed_list = [np.random.randint(1000000000) for _ in range(args.num_trial)]
###############################

###############################
# initialize data
def make_dataloader(args):
    if args.data == "cifar":
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # Subset data to local agent
        num_samples = 50000 // size
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in
                range(int(rank * num_samples), int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 10000 // size
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transform),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
    elif args.data == "mnist":
        # Create transform for data
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

        # Subset data to local agent
        num_samples = 60000 // size
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                        transform=transform),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                        transform=transform),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in
                range(int(rank * num_samples), int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 10000 // size
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
    elif args.data == 'a9a':
        # Subset data to local agent
        num_samples = 32561 // size
        train_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in
                range(int(rank * num_samples), int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 16281 // size
        test_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=False),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
            [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
    elif args.data == 'miniboone':
        # Subset data to local agent
        num_samples = 100000 // size
        train_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Load data to be used to compute full gradient with neighbors
        optimality_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=True),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in
                    range(int(rank * num_samples),
                        int((rank + 1) * num_samples))]))  # Difference is in number of samples!!

        # Load the testing data
        num_test = 30064 // size
        test_loader = torch.utils.data.DataLoader(
            BinaryDataset('data', args.data, train=False),
            batch_size=num_test, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_test), int((rank + 1) * num_test))]))
        
    return train_loader, optimality_loader, test_loader

train_loader, optimality_loader, test_loader = make_dataloader(args)
###############################

###############################
# start the training
for trial in range(args.num_trial):
    # Load communication matrix
    # TODO: write a code to generate this matrix for any size
    mixing_matrix = torch.tensor(np.load(f'mixing_matrices/{args.comm_pattern}_{size}.dat', allow_pickle=True))

    # Print training information
    if rank == 0:
        opening_statement = f' {args.algorithm} on {args.data}, trial {trial+1} '
        print(f"\n{'#' * 75}")
        print('\n' + opening_statement.center(75, ' '))
        print(
            f'[GRAPH INFO] {size} agents | connectivity = {args.comm_pattern} | rho = {torch.sort(torch.linalg.eigvals(mixing_matrix).real)[0][size - 2].item()}')
        print(f'[TRAINING INFO] mini-batch = {args.mini_batch} | learning rate = {args.lr}\n')
        print(f"{'#' * 75}\n")

    # Barrier before training
    comm.Barrier()

    # Declare and train!
    method = args.algorithm
    local_params = {'lr': args.lr, 'mini_batch': args.mini_batch, 'report': args.report, 'model': args.model, 'device': args.device,
                    'step_type': args.step_type, 'l1': args.l1, 'seed': args.init_seed_list[trial], 'data': args.data}
    solver = solver_dict[method](local_params, mixing_matrix, train_loader)
    algo_time = solver.solve(args.updates, optimality_loader, test_loader)

    # Save the information
    # collect all results in a common folder
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/'))
    except:
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/'))
    except:
        pass
    try:
        os.mkdir(os.path.join(os.getcwd(), f'results/{args.data}'))
    except:
        pass
    path = os.path.join(os.getcwd(), f'results/{args.data}')

    # Save information via np
    if rank == 0:
        all_results = [solver.testing_loss, solver.testing_accuracy, solver.training_loss, solver.training_accuracy,\
                    solver.testing_loss_local, solver.testing_accuracy_local, solver.training_loss_local, solver.training_accuracy_local,\
                    solver.total_optimality, solver.consensus_violation, solver.norm_hist, solver.iterate_norm_hist, solver.total_time,\
                    solver.communication_time, solver.compute_time, solver.nnz_at_avg, solver.avg_nnz]
        all_results = np.array(all_results, dtype=object)
        if args.step_type == 'diminishing':
            save_path = f'{path}/{method}_t_{trial+1}_{args.comm_pattern}_{args.mini_batch}_{args.updates}_lr_{args.lr}.npy'
        else:
            save_path = f'{path}/{method}_t_{trial+1}_{args.comm_pattern}_{args.mini_batch}_{args.updates}_lr_{args.lr}_step_{args.step_type}.npy'
        np.save(save_path, all_results)
    # Barrier at end so all agents stop this script before moving on
    comm.Barrier()
###############################