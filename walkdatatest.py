import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from os import walk
# import dill as pickle
import pickle as pkl

mypath = '/home/azureuser/cloudfiles/code/Users/esd27/piedatawalk'
_, _, filenames = next(walk(mypath))

with open('/home/azureuser/cloudfiles/code/Users/esd27/piedatawalk/' + filenames[0], "rb") as f:
    package = pkl.load(f)
    array = package[0]
    truth = torch.tensor(package[1]).float()
    length = len(array)
    query = array[length-1]

print("Representations:")
for rep in array:
    print(rep)
print("Query:")
print(query)
print('Truth:')
print(truth)