import torch
import code
import os
from dataset_definition import *

def test_load_dataset(name):
    path = os.path.join(r"C:\Users\Ben\Desktop\VSCodeCoding\FDUInternship\saved_datasets", name)
    if not os.path.exists(path):
        print("File path does not exist")
        return
    torch.load(path, weights_only=False)
    print("Dataset loaded successfully!")

code.interact(locals=locals())