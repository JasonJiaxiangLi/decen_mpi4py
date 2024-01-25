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

class DSGT(Base):
    '''
    Class for solving decentralized nonconvex consensus problems 
    using descentralized sthocastic gradient tracking
    '''

    def __init__(self):
        super().__init__()

    def onestep_update(self):
        ##################################################
        # Perform the algorithm updates
        # Communicate with neighbors
        comm.Barrier()
        comm_time1 = self.communicate_with_neighbors()
        comm.Barrier()

        # Time part of computation time
        time_i = time.time()

        # UPDATE WEIGHTS
        self.weights = self.regularizer.forward(
            [self.weights[k].detach().clone() - self.lr * self.Y[k].detach().clone() for k in
                range(self.num_params)], self.lr * self.l1)

        # STOP TIME FOR COMPUTING
        int_time1 = time.time()

        # Save pre communication weights for computing nnz
        pre_comm_weights = [self.weights[k].detach().clone() for k in range(self.num_params)]

        # Communicate the variables
        comm.Barrier()
        comm_time1 = self.communicate_with_neighbors()
        comm.Barrier()

        # STOP TIME FOR COMPUTING
        int_time2 = time.time()

        # UPDATE THE GRADIENTS
        self.grads = self.get_grads(self.weights)

        # END TIME
        int_time2_end = time.time()

        # Communicate Y
        comm.Barrier()
        comm_time_2 = self.communicate_y_with_neighbors()
        comm.Barrier()

        # STOP TIME FOR COMPUTING
        int_time3 = time.time()

        # UPDATE THE Y VARIABLE
        # this is the main difference of DSGT to DSGD
        self.Y = [self.Y[k] + self.grads[k] - self.prev_grads[k] for k in range(self.num_params)]

        # SAVE GRADIENTS
        self.prev_grads = [self.grads[pa].detach().clone() for pa in range(len(self.grads))]

        # End computation time
        time_i_end = time.time()

        # Save times
        comp_time = round(time_i_end - int_time3 + int_time2_end - int_time2 + int_time1 - time_i, 4)
        comm_time = comm_time_2 + comm_time1
        ##################################################
        return comp_time, comm_time, pre_comm_weights