# for (arr_name, arr) in arrays:
#         torch.save(arr, f"{directory}/{arr_name}_final_{index}")

from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import os
import math


import torch

directory = "/home/xuhui.li/Downloads/edge-of-stability-github/my/results/cifar10-1k/fc-tanh/seed_0/mse/adam/lr_0.01_adam"  # Replace with the actual directory
index = 42  # Replace with the actual index

# Example: Reading a tensor saved with torch.save
arr_name = "your_array"  # Replace with the actual array name

for index in range(100):
    file_path = f"{directory}/train_loss_final_{index}"
    file_path2 = f"{directory}/max_loss_final_{index}"

    # Load the tensor from the saved file
    loaded_tensor = torch.load(file_path)
    loaded_tensor2 = torch.load(file_path2)

    # Now, 'loaded_tensor' contains the tensor loaded from the saved file
    print(f"Size of loaded_tensor: {loaded_tensor.size()}")

    # Iterate over elements of the tensor
    # for i in range(loaded_tensor.size(0)):
    print(loaded_tensor, loaded_tensor2)
