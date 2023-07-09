from PIL import Image 
import torch
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader, Sampler
from matplotlib import pyplot as plt 
import math 



# Use SubsetSampler to set the number of pictures for each epoch 
class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# Get the Single Picture to train
class MyDataSet(torch.utils.data.Dataset): 
    def __init__(self,root, filename, transform=None): 
        
        self.root = root
        img = Image.open(self.root+filename).convert('RGB')
        self.img = img
        self.transform = transform 
        
    def __getitem__(self, index): 
    
        if self.transform is not None: 
            img = self.transform(self.img) 
        
        return img  
    
    def __len__(self): 
        return 100000


   