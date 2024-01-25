#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test the decentralized stochastic gradient descent

Papers:
    [1] https://arxiv.org/pdf/1909.02712v2.pdf
    [2] https://openreview.net/pdf?id=CmI7NqBR4Ua
    [3] https://ieeexplore.ieee.org/abstract/document/8755807

"""

# Import packages
from __future__ import print_function
import argparse
import os
import sys
import time
import torch
import numpy
import math
from mpi4py import MPI
from torch.optim import SGD
from torchvision import datasets, transforms

# Import custom classes
from models.mlp import MLP
from models.lenet import LENET
from helpers.l1_regularizer import L1
from helpers.replace_weights import Opt
from helpers.custom_data_loader import BinaryDataset

# Set up MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class Base:
    '''
    Class for solving decentralized nonconvex consensus problems
    This is served as a base algorithm

    :param: local_params = DICT of parameters for training
    :param: mixing_matrix = NxN torch float containing weights for communication
    :param: training_data = torch.utils.data.Dataloader
    :param: init_seed
    '''

    def __init__(self, args, mixing_matrix, training_data, init_seed):
        local_params = {'lr': args.lr, 'mini_batch': args.mini_batch, 'report': args.report,
                        'step_type': args.step_type, 'l1': args.l1, 'seed': init_seed}
        # Get the information about neighbor communication:
        # First, we extract the number of nodes and double check
        # this value is the same as the size of the MPI world
        # Second, we extract thr row of the mixing matrix corresponding to this agent
        # and save the weights
        self.mixing_matrix = mixing_matrix.float()
        self.num_nodes = self.mixing_matrix.shape[0]
        if self.num_nodes != size:
            sys.exit(f"Cannot match MPI size {size} with mixing matrix of shape {self.num_nodes}. ")
        self.peers = torch.where(self.mixing_matrix[rank, :] != 0)[0].tolist()
        self.peers.remove(rank)
        self.peer_weights = self.mixing_matrix[rank, self.peers].tolist()
        self.my_weight = self.mixing_matrix[rank, rank].item()

        ##################################################
        # Parse the training parameters:
        #
        # model = 'bilinear', 'lenet', or 'fc' for architecture type (STR)
        # step_type = 'constant' or 'diminishing' for learning rate type (STR)
        # alpha = learning rate (FLOAT)
        # beta = variance reduction term (FLOAT)
        # beta1 = variance reduction term in denominator (FLOAT)
        # k0 = parameter in alpha (INT)
        # mini_batch = batch size (INT)
        # init_batch = initial batch size (INT)
        # l1 = regularization coefficient (FLOAT)
        # report = how often to report stationarity, test acc, etc. (INT)
        ##################################################
        if 'step_type' in local_params:
            self.step_type = local_params['step_type']
        else:
            self.step_type = 'diminishing'
        if 'lr' in local_params:
            self.lr_base = local_params['lr']
        else:
            self.lr_base = 1.0
        if 'mini_batch' in local_params:
            self.mini_batch = int(local_params['mini_batch'])
        else:
            self.mini_batch = 128
        if 'l1' in local_params:
            self.l1 = local_params['l1']
        else:
            self.l1 = 0.0
        if 'report' in local_params:
            self.report = local_params['report']
        else:
            self.report = 100

            # Fix stepsize to be constant, IF that is specified
        if self.step_type == 'constant':
            self.lr = self.lr_base
        else:
            pass

        # Get the CUDA device and save the data loader to be easily reference later
        # self.device = torch.device(f'cuda:{rank % size}')
        self.device = torch.device(f'cpu:{rank % size}')
        self.data_loader = training_data

        # Initialize the models
        # We either have the MLP or we have LENET
        if args.data in ['a9a', 'miniboone']:
            self.model = MLP(self.data_loader.dataset.data.shape[1], 64, 2).to(self.device)

        elif args.data in ['mnist', 'cifar']:
            self.model = LENET(10).to(self.device)

        else:
            sys.exit(f"[ERROR] To use a new dataset/architecture, add the dataset to the data folder and incorporate the"
                     f"model here using \'self.model = <your_model>.to(self.device)\'.")

        # Initialize the updating weights rule and the training loss function
        self.replace_weights = Opt(self.model.parameters(), lr=0.1)
        self.training_loss_function = torch.nn.NLLLoss(reduction='mean')

        # Initialize the l1 regularizer
        self.regularizer = L1(self.device)

        # initialize variables
        for param in self.model.parameters():
            torch.manual_seed(local_params['seed'])
            param.data = torch.randn_like(param.data)

        self.weights = [param.data for param in self.model.parameters()]

        # Save number of parameters
        self.num_params = len(self.weights)

        # initialize all parameter grads and Y
        ## modify this function to include all variables you add
        self.initial_grads()

        # Allocate space for relevant report values: consensus, gradient,
        # iterate norm, number non-zeros, training/testing acc, compute time, etc.
        self.consensus_violation = []
        self.norm_hist = []
        self.total_optimality = []
        self.iterate_norm_hist = []
        self.nnz_at_avg = []
        self.avg_nnz = []
        self.testing_loss = []
        self.testing_accuracy = []
        self.training_loss = []
        self.training_accuracy = []
        self.testing_loss_local = []
        self.testing_accuracy_local = []
        self.training_loss_local = []
        self.training_accuracy_local = []
        self.compute_time = []
        self.communication_time = []
        self.total_time = []

    def initial_grads(self):
        """"Initialize Y and use Y to communicate with neighbors"""
        self.grads = self.get_grads(self.weights)
        self.Y = [self.grads[k].detach() for k in range(self.num_params)]
        self.prev_grads = [self.grads[k].detach() for k in range(self.num_params)]

    def solve(self, outer_iterations, training_data_full_sample, testing_data):
        '''Solve the global problem'''

        # Communicate Y to have first gradient tracking term
        comm.Barrier()
        _ = self.communicate_y_with_neighbors()
        comm.Barrier()

        # Barrier
        comm.Barrier()

        ##################################################
        # Save initial errors for fair comparison across methods
        avg_weights = self.get_average_param(self.weights)
        cons, norm, total, var_norm, nnz_at_avg, avg_nnz = self.compute_optimality_criteria(avg_weights, self.weights,
                                                                                            training_data_full_sample)
        self.consensus_violation.append(cons)
        self.norm_hist.append(norm)
        self.total_optimality.append(total)
        self.iterate_norm_hist.append(var_norm)
        self.nnz_at_avg.append(nnz_at_avg)
        self.avg_nnz.append(avg_nnz)

        # TEST ACCURACY ON TRAINING SET
        train_loss, train_acc = self.test(avg_weights, self.data_loader)
        self.training_loss.append(train_loss)
        self.training_accuracy.append(train_acc)

        # TEST ACCURACY ON TEST SET
        test_loss, test_acc = self.test(avg_weights, testing_data)
        self.testing_loss.append(test_loss)
        self.testing_accuracy.append(test_acc)

        # TEST ACCURACY ON TRAINING SET AT LOCAL
        train_loss_local, train_acc_local = self.test(self.weights, self.data_loader, mode='local')
        self.training_loss_local.append(train_loss_local)
        self.training_accuracy_local.append(train_acc_local)

        # TEST ACCURACY ON TEST SET AT LOCAL
        test_loss_local, test_acc_local = self.test(self.weights, testing_data, mode='local')
        self.testing_loss_local.append(test_loss_local)
        self.testing_accuracy_local.append(test_acc_local)
        ##################################################

        # Time the entire algorithm
        t0 = time.time()

        # Barrier communication at beginning of run to force agents to start at the same time
        comm.Barrier()

        # Loop over algorithm updates
        for i in range(outer_iterations):

            self.update_learning_rate(i, outer_iterations)

            # this is the main update
            ## Replace this function with the new updates!
            comp_time, comm_time, pre_comm_weights = self.onestep_update()

            # Barrier at the end of update for extreme safety
            comm.Barrier()

            # Save values at report interval
            if i % self.report == 0:

                # Save the first errors using the average value - so all agents are compared fairly
                avg_weights = self.get_average_param(self.weights)
                cons, norm, total, var_norm, nnz_at_avg, avg_nnz = self.compute_optimality_criteria(avg_weights,
                                                                                                    self.weights,
                                                                                                    training_data_full_sample,
                                                                                                    pre_comm_weights)
                self.consensus_violation.append(cons)
                self.norm_hist.append(norm)
                self.total_optimality.append(total)
                self.iterate_norm_hist.append(var_norm)
                self.nnz_at_avg.append(nnz_at_avg)
                self.avg_nnz.append(avg_nnz)

                # TEST ACCURACY ON TRAINING SET
                train_loss, train_acc = self.test(avg_weights, self.data_loader)
                self.training_loss.append(train_loss)
                self.training_accuracy.append(train_acc)

                # TEST ACCURACY ON TEST SET
                test_loss, test_acc = self.test(avg_weights, testing_data)
                self.testing_loss.append(test_loss)
                self.testing_accuracy.append(test_acc)

                # TEST ACCURACY ON TRAINING SET AT LOCAL
                train_loss_local, train_acc_local = self.test(self.weights, self.data_loader, mode='local')
                self.training_loss_local.append(train_loss_local)
                self.training_accuracy_local.append(train_acc_local)

                # TEST ACCURACY ON TEST SET AT LOCAL
                test_loss_local, test_acc_local = self.test(self.weights, testing_data, mode='local')
                self.testing_loss_local.append(test_loss_local)
                self.testing_accuracy_local.append(test_acc_local)

                # Print relevant information
                if rank == 0:
                    # First iteration, print headings, then print the values
                    if i == 0:
                        print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<15} | {:<15} | {:<12} | {:<6}".format("Iteration", "Epoch",
                                                                                                  "Stationarity",
                                                                                                  "Train (L / A)",
                                                                                                  "Test (L / A)",
                                                                                                  "Train (L / A) L",
                                                                                                  "Test (L / A) L",
                                                                                                  "Avg Density",
                                                                                                  "Time"))
                    print("{:<10} | {:<7} | {:<13} | {:<15} | {:<15} | {:<15} | {:<15} | {:<12} | {:<6}".format(i,
                                                                     round((i * self.mini_batch) / (self.data_loader.dataset.data.shape[0] // size), 2),
                                                                     round(total, 4),
                                                                     f"{round(train_loss, 4)} / {round(train_acc, 2)}",
                                                                     f"{round(test_loss, 4)} / {round(test_acc, 2)}",
                                                                     f"{round(train_loss_local, 4)} / {round(train_acc_local, 2)}",
                                                                     f"{round(test_loss_local, 4)} / {round(test_acc_local, 2)}",
                                                                     round(avg_nnz, 6),
                                                                     round(time.time() - t0, 1)))

            # Append timing information for each iteration
            self.compute_time.append(comp_time)
            self.communication_time.append(comm_time)
            self.total_time.append(comp_time + comm_time)

        ##################################################
        # End total training time
        t1 = time.time() - t0
        if rank == 0:
            closing_statement = f' Training finished '
            print('\n' + closing_statement.center(50, '-'))
            print(f'[TOTAL TIME] {round(t1, 2)}')

        # Return the training time
        return t1
    
    def update_learning_rate(self, i, outer_iterations):
        """TODO: using learning rate schedulers"""
        # Step step-sizes
        if self.step_type == 'constant':
            self.lr = self.lr_base * math.sqrt(self.num_nodes / outer_iterations)

        # Diminishing step-sizes
        else:
            self.lr = self.lr_base / math.pow(i + 1, 1 / 2)

    def onestep_update(self):
        pass

    def communicate_y_with_neighbors(self):
        '''Update the gradient tracking'''

        # Time the Y communication
        time0 = MPI.Wtime()

        # Loop over parameters doing and Isend/Irecv
        for pa in range(self.num_params):

            # DEFINE VARIABLE TO SEND
            send_data = self.Y[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=((len(self.peers),) + self.Y[pa].shape), dtype=numpy.float32)

            # SET UP REQUESTS TO INSURE CORRECT SENDS/RECVS
            recv_request = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]

            # SEND THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Send the data
                recv_request[ind + len(self.peers)] = comm.Isend(send_data, dest=peer_id)

            # RECEIVE THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Receive the data
                recv_request[ind] = comm.Irecv(recv_data[ind, :], source=peer_id)

            # HOLD UNTIL ALL COMMUNICATIONS COMPLETE
            MPI.Request.waitall(recv_request)

            # SCALE CURRENT WEIGHTS
            self.Y[pa] = self.my_weight * self.Y[pa]

            # Update global variables
            for ind in range(len(self.peers)):
                self.Y[pa] += (self.peer_weights[ind] * torch.tensor(recv_data[ind, :]).to(self.device))

        return round(MPI.Wtime() - time0, 4)

    def communicate_with_neighbors(self):

        # Time this communication
        time0 = MPI.Wtime()

        # Loop over all of the variables
        for pa in range(self.num_params):

            # DEFINE VARIABLE TO SEND
            send_data = self.weights[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=((len(self.peers),) + self.weights[pa].shape), dtype=numpy.float32)

            # SET UP REQUESTS TO INSURE CORRECT SENDS/RECVS
            recv_request = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]

            # SEND THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Send the data
                recv_request[ind + len(self.peers)] = comm.Isend(send_data, dest=peer_id)

            # RECEIVE THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Receive the data
                recv_request[ind] = comm.Irecv(recv_data[ind, :], source=peer_id)

            # HOLD UNTIL ALL COMMUNICATIONS COMPLETE
            MPI.Request.waitall(recv_request)

            # SCALE CURRENT WEIGHTS
            self.weights[pa] = self.my_weight * self.weights[pa]

            # Update global variables
            for ind in range(len(self.peers)):
                self.weights[pa] += (self.peer_weights[ind] * torch.tensor(recv_data[ind, :]).to(self.device))

        return round(MPI.Wtime() - time0, 4)

    def get_average_param(self, list_of_params):
        '''Perform ALLREDUCE of neighbor parameters'''

        # Save information to blank list
        output_list_of_parameters = [None] * len(list_of_params)

        # Loop over the parameters
        for pa in range(self.num_params):

            # Prep send and receive to be numpy arrays
            send_data = list_of_params[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=(list_of_params[pa].shape), dtype=numpy.float32)

            # Barriers and note that the allreduce operations is summation!
            comm.Barrier()
            comm.Allreduce(send_data, recv_data)
            comm.Barrier()

            # Save information by dividing by number of agents and converting to tensor
            output_list_of_parameters[pa] = (1 / self.num_nodes) * torch.tensor(recv_data).to(self.device)

        return output_list_of_parameters

    def get_grads(self, current_weights):
        '''Get a local gradient'''

        # Set model to training mode
        self.model.train()

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Print errors
            torch.autograd.set_detect_anomaly(True)

            # Convert data to CUDA if possible
            data, target = data.to(self.device).float(), target.to(self.device).long()

            # Zero out gradients
            self.replace_weights.step(current_weights, self.device)
            self.replace_weights.zero_grad()
            # Forward pass of the model
            out1 = self.model(data)
            loss1 = (1 / self.num_nodes) * self.training_loss_function(out1, target)
            # Compute the gradients
            loss1.backward()

            # Update D
            grads = [p.grad.data.detach().clone() for ind, p in enumerate(self.model.parameters())]
            break

        return grads

    def compute_optimality_criteria(self, avg_weights, local_weights, training_data_full_sample, pre_comm_weights=None):
        '''
        Compute the relevant metrics for this problem

        :param avg_weights: LIST of average weights
        :param local_weights: LIST of local weights
        :param training_data_full_sample: data loader with full gradient size
        :return:
        '''

        # Compute consensus for this agent
        local_violation = sum([numpy.linalg.norm(
            local_weights[i].cpu().numpy().flatten() - avg_weights[i].cpu().numpy().flatten(), ord=2) ** 2 for i in
                               range(len(local_weights))])

        # Compute the norm of the iterate to save in case consensus is large
        avg_weight_norm = sum([numpy.linalg.norm(avg_weights[i].cpu().numpy().flatten(), ord=2) ** 2 for i in
                               range(len(avg_weights))])

        # Compute the gradient at the average solution on this dataset:
        # 1. Replace the model params
        # 2. Forward pass, backward pass to have gradient
        # 3. Compute the stationarity violation
        # 4. MUST SCALE: total number of samples is (N * num_local) samples. Since `get_average_param` divides by N
        # the loss function here must be scaled only by (1 / num_local)
        loss_function = torch.nn.NLLLoss(reduction='sum')
        coef = 1. / (len(training_data_full_sample.dataset) // size)

        self.replace_weights.step(avg_weights, self.device)
        self.model.train()
        grads = [torch.zeros(size=p.shape).to(self.device) for p in self.model.parameters()]
        for batch_idx, (data, target) in enumerate(training_data_full_sample):
            # Print errors (just in case) and zero out the gradient
            torch.autograd.set_detect_anomaly(True)
            self.replace_weights.zero_grad()
            data, target = data.to(self.device).float(), target.to(self.device).long()

            # Forward and backward pass of the model; scale by (1 / N) to line up with average
            out = self.model(data)
            loss = coef * loss_function(out, target)
            loss.backward()

            # Save gradients
            grads = [grads[ind] + p.grad.data.detach().clone().to(self.device) for ind, p in
                     enumerate(self.model.parameters())]

        # Get the average gradient by doing all_reduce and then compute the stationarity violation at the average point
        avg_grads = self.get_average_param(grads)
        stationarity1 = self.regularizer.forward([avg_weights[pa] - avg_grads[pa] for pa in range(self.num_params)],
                                                 self.l1)
        stationarity = numpy.concatenate([avg_weights[pa].detach().cpu().numpy().flatten()
                                          - stationarity1[pa].detach().cpu().numpy().flatten() for pa in
                                          range(self.num_params)])
        global_norm = numpy.linalg.norm(stationarity, ord=2) ** 2

        # Before sending, also get then number of non-zeros for this agent and this average
        if pre_comm_weights is None:
            _, local_nnz_ratio = self.regularizer.number_non_zeros(local_weights)
            _, nnz_at_average = self.regularizer.number_non_zeros(avg_weights)
        else:
            _, local_nnz_ratio = self.regularizer.number_non_zeros(pre_comm_weights)
            _, nnz_at_average = self.regularizer.number_non_zeros(avg_weights)

        # Perform all-reduce to have sum of local violations, i.e. Frobenius norm of consensus
        array_to_send = numpy.array([local_violation, local_nnz_ratio])
        recv_array = numpy.empty(shape=array_to_send.shape)
        comm.Barrier()
        comm.Allreduce(array_to_send, recv_array)
        comm.Barrier()

        # return consensus, gradient, total optimality, iterate history,
        # local number non-zeros, number nonzeros at everate, and average number of nonzeros
        return recv_array[0], global_norm, recv_array[0] + global_norm, avg_weight_norm, \
               nnz_at_average, (1 / size) * recv_array[1]

    def test(self, weights, testing_data, mode='global'):
        '''Test the data using the average weights'''

        self.replace_weights.zero_grad()
        self.replace_weights.step(weights, self.device)
        self.model.eval()

        # Create separate testing loss for testing data
        loss_function = torch.nn.NLLLoss(reduction='sum')

        # Allocate space for testing loss and accuracy
        test_loss = 0
        correct = 0

        # Do not compute gradient with respect to the testing data
        with torch.no_grad():
            # Loop over testing data
            for data, target in testing_data:
                # Use CUDA
                data, target = data.to(self.device).float(), target.to(self.device).long()

                # Evaluate the model on the testing data
                output = self.model(data)
                test_loss += loss_function(output, target).item()

                # Gather predictions on testing data
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute number of testing data points
        num_test_points = int(len(testing_data.dataset) / size)

        # We have two modes of reporting data:
        # 1. We use the AVG weights on all of the data
        # 2. We use the local weights on the local data and THEN compute average
        if mode == 'global':
            # PERFORM ALL REDUCE TO HAVE AVERAGE
            array_to_send = numpy.array([correct, num_test_points, test_loss])
            recv_array = numpy.empty(shape=array_to_send.shape)

            # Barrier
            comm.Barrier()
            comm.Allreduce(array_to_send, recv_array)
            comm.Barrier()

            # Save loss and accuracy
            test_loss = recv_array[2] / recv_array[1]
            testing_accuracy = 100 * recv_array[0] / recv_array[1]

        # Compue local information and then average
        elif mode == 'local':
            # PERFORM ALL REDUCE TO HAVE AVERAGE
            correct /= num_test_points
            test_loss /= num_test_points
            array_to_send = numpy.array([correct, test_loss])
            recv_array = numpy.empty(shape=array_to_send.shape)

            # Barrier
            comm.Barrier()
            comm.Allreduce(array_to_send, recv_array)
            comm.Barrier()

            # Save loss and accuracy
            test_loss = recv_array[1] / size
            testing_accuracy = 100 * recv_array[0] / size
        else:
            sys.exit(f"[ERROR] _ {mode} _ is not a vaild report metric; choose from \'local\' or \'global\' [ERROR]")

        return test_loss, testing_accuracy