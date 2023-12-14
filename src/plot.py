# import pickle


# path = "/home/siem.hadish/Desktop/edge-of-stability-github/data/0"
# with open(path, 'rb') as file:
#     data = pickle.load(file)
print("tghe desktop one")
# import torch
import matplotlib.pyplot as plt
import torch
import os

# Assuming 'extracted_directory' is the directory where you extracted the ZIP file
extracted_directory = "/home/xuhui.li/Downloads/edge-of-stability-github/my/results/cifar10-5k/fc-tanh/seed_0/mse/gd/lr_0.01/"
extracted_files =  ['train_acc'] #['eigs', 'iterates', 'train_loss', 'test_loss', 'train_acc', 'test_acc']

for file_name in extracted_files:
    file_path = os.path.join(extracted_directory, file_name + '_final')  # Assuming .pt extension
    data = torch.load(file_path)
    # 'data' now contains the contents of the file. Process as needed.


import pandas as pd
df = pd.DataFrame(data)
df.plot()
plt.show()