
import os
from os import walk

mypath = '/home/azureuser/cloudfiles/code/Users/esd27'

_, _, filenames = next(walk(mypath))

for f in filenames:
    if f.endswith(".npy"):
        os.remove('/home/azureuser/cloudfiles/code/Users/esd27/' + f)