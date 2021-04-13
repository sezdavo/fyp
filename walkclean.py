import os
from os import walk
import pickle as pkl

mypath = '/home/azureuser/cloudfiles/code/Users/esd27/piedatawalk'
_, _, filenames = next(walk(mypath))

for f in filenames:
    if f.endswith(".p"):
        with open('/home/azureuser/cloudfiles/code/Users/esd27/piedatawalk/' + f, "rb") as item:
            package = pkl.load(item)
            array = package[0]
            if len(array) != 30:
                print('found bad item')
                os.remove('/home/azureuser/cloudfiles/code/Users/esd27/piedatawalk/' + f)
                