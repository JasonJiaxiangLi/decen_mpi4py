#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test the descentralized stohcastic normalized averaged gradient tracking

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
from torchvision import datasets, transforms

# Import custom classes
from algorithms.base import Base
from models.mlp import MLP
from models.lenet import LENET
from helpers.l1_regularizer import L1
from helpers.replace_weights import Opt
from helpers.custom_data_loader import BinaryDataset

# Set up MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class DNASA(Base):
    '''
    Class for solving decentralized nonconvex consensus problems 
    using descentralized sthocastic normalized averaged gradient tracking
    '''

    def initial_grads(self):
        self.grads = self.get_grads(self.weights)
        
        # Initialize local y, z, u, and previous grads
        self.Y = [self.weights[k].detach() for k in range(self.num_params)]
        self.Z = [torch.zeros_like(self.weights[k]) for k in range(self.num_params)]
        self.U = [self.grads[k].detach() for k in range(self.num_params)]
        self.prev_grads = [self.grads[k].detach() for k in range(self.num_params)]

    def update_learning_rate(self, i, outer_iterations):
        """TODO: using learning rate schedulers"""
        # constant step-sizes
        if self.step_type == 'constant':
            self.alpha = self.alpha_base * math.sqrt(self.num_nodes / outer_iterations)
            self.lr = self.lr_base * (self.num_nodes)**(1/4) / (outer_iterations) ** (3/4)
        # Diminishing step-sizes
        else:
            self.alpha = self.alpha_base * math.sqrt(self.num_nodes / (i + 1))
            self.lr = self.lr_base * (self.num_nodes)**(1/4) / (i + 1) ** (3/4)

    def onestep_update(self):
        time_i = time.time()
        ##################################################
        # Update local y via prox
        Y = [0.0] * self.num_params
        for k in range(self.num_params):
            temp = self.Z[k].detach().clone()
            if torch.linalg.norm(temp) <= sys.float_info.epsilon:
                Y[k] = self.weights[k].detach().clone()
            else:
                Y[k] = self.weights[k].detach().clone() - self.lr * temp / torch.linalg.norm(temp)
        self.Y = self.regularizer.forward(Y, self.lr * self.l1)

        # Obtain local grads
        self.grads = self.get_grads(self.weights)

        # Update local x via moving average
        # self.weights = [(1 - self.alpha) * self.weights[k].detach().clone() + \
        #                         self.alpha * self.Y[k].detach().clone() for k in range(self.num_params)]
        self.weights = [self.Y[k].detach().clone() for k in range(self.num_params)]

        # Update local z via moving average
        self.Z = [(1 - self.alpha) * self.Z[k].detach().clone() + \
                                self.alpha * self.grads[k] for k in range(self.num_params)]

        # Update u via gradient tracking
        self.U = [self.U[k] + self.grads[k] - self.prev_grads[k] for k in range(self.num_params)]

        # Save pre grads for gradient tracking
        self.prev_grads = [self.grads[pa].detach().clone() for pa in range(len(self.grads))]
        ##################################################


        # STOP TIME FOR COMPUTING
        time_i_end = time.time()
        comm_time1 = 0
        ##################################################
        # Communication
        for _ in range(self.comm_round):
            # Communicate X, U, and Z
            comm.Barrier()
            comm_time1 += self.communicate_with_neighbors()
            comm.Barrier()
        ##################################################

        # Save pre communication weights for computing nnz
        pre_comm_weights = [self.weights[k].detach().clone() for k in range(self.num_params)]

        # Save times
        comp_time = round(time_i_end - time_i, 4)
        comm_time = comm_time1
        ##################################################
        return comp_time, comm_time, pre_comm_weights

    def communicate_with_neighbors(self):
        # Time this communication
        time0 = MPI.Wtime()

        # Loop over all of the variables
        for pa in range(self.num_params):

            # DEFINE VARIABLE TO SEND
            send_data = self.weights[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=((len(self.peers),) + self.weights[pa].shape), dtype=numpy.float32)
            send_data_z = self.Z[pa].cpu().detach().numpy()
            recv_data_z = numpy.empty(shape=((len(self.peers),) + self.Z[pa].shape), dtype=numpy.float32)
            send_data_u = self.U[pa].cpu().detach().numpy()
            recv_data_u = numpy.empty(shape=((len(self.peers),) + self.U[pa].shape), dtype=numpy.float32)

            # SET UP REQUESTS TO INSURE CORRECT SENDS/RECVS
            recv_request = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]
            recv_request_z = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]
            recv_request_u = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]


            # SEND THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Send the data
                recv_request[ind + len(self.peers)] = comm.Isend(send_data, dest=peer_id)
                recv_request_z[ind + len(self.peers)] = comm.Isend(send_data_z, dest=peer_id)
                recv_request_u[ind + len(self.peers)] = comm.Isend(send_data_u, dest=peer_id)


            # RECEIVE THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Receive the data
                recv_request[ind] = comm.Irecv(recv_data[ind, :], source=peer_id)
                recv_request_z[ind] = comm.Irecv(recv_data_z[ind, :], source=peer_id)
                recv_request_u[ind] = comm.Irecv(recv_data_u[ind, :], source=peer_id)


            # HOLD UNTIL ALL COMMUNICATIONS COMPLETE
            MPI.Request.waitall(recv_request)
            MPI.Request.waitall(recv_request_z)
            MPI.Request.waitall(recv_request_u)

            # SCALE CURRENT WEIGHTS
            self.weights[pa] = self.my_weight * self.weights[pa]
            self.Z[pa] = self.my_weight * self.Z[pa]
            self.U[pa] = self.my_weight * self.U[pa]

            # Update global variables
            for ind in range(len(self.peers)):
                self.weights[pa] += (self.peer_weights[ind] * torch.tensor(recv_data[ind, :]).to(self.device))
                self.Z[pa] += (self.peer_weights[ind] * torch.tensor(recv_data_z[ind, :]).to(self.device))
                self.U[pa] += (self.peer_weights[ind] * torch.tensor(recv_data_u[ind, :]).to(self.device))

        return round(MPI.Wtime() - time0, 4)