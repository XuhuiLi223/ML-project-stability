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
import matplotlib.pyplot as plt

import torch

directory = "/home/siem.hadish/Desktop/edge-of-stability-github/my/results/13th/cifar10-5k/fc-tanh/seed_0/mse/gd/lr_0.01/"  # Replace with the actual directory
index = 42  # Replace with the actual index

# Example: Reading a tensor saved with torch.save
arr_name = "your_array"  # Replace with the actual array name
ave_uni = []
for index in range(100):
    file_path = f"{directory}/sample_loss_final_{index}"
    file_path2 = f"{directory}/max_loss_final_{index}"

    # Load the tensor from the saved file
    loaded_tensor = torch.load(file_path)
    loaded_tensor2 = torch.load(file_path2)

    # loaded_tens_numpy = loaded_tensor.numpy()
    # loaded_tens_numpy2 = loaded_tensor2.numpy()

    # Now, 'loaded_tensor' contains the tensor loaded from the saved file
    # print(f"Size of loaded_tensor: {loaded_tensor.size()}")

    # Iterate over elements of the tensor
    # for i in range(loaded_tensor.size(0)):
    ave_uni.append([abs(float(loaded_tensor[0])), loaded_tensor2])
    # print(abs(float(loaded_tensor[0])), loaded_tensor2, index)
    # print("printing Numpy values")
    # print(loaded_tens_numpy, loaded_tens_numpy2)
np.random.shuffle(ave_uni)
# print(ave_uni)

ave_uni_array = np.array(ave_uni)

# Extracting the first and second values
first_values = ave_uni_array[:, 0]
second_values = ave_uni_array[:, 1]
average_first_line = np.mean(first_values)
max_second_line = np.max(second_values)

print("Average of the first column:", average_first_line)
print("Maximum of the second column:", max_second_line)

# Creating indices for x-axis
indices = np.arange(len(ave_uni_array))

a = (average_first_line, max_second_line)

# Specify the filename
filename = "cifar10-5k-fc-tanh-sgfin.txt"

# Saving the tuple to a file
with open(filename, 'w') as file:
    file.write(str(a))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(indices, first_values, label='Average stability of Z_i')
plt.plot(indices, second_values, label='Uniform stabiliity per hybrid sample S_i')

# Adding title and labels
plt.title('Line Plot of Average an uniform stabilities')
plt.xlabel('Z-Index')
plt.ylabel('MSE difference')
plt.legend()
plt.savefig(f'{directory}/my_plot.png', dpi=600)  # 
# Show the plot
plt.show()
