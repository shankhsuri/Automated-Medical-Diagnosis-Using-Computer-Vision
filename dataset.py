import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torchvision
import numpy as np
import torchvision.models as models

# Get the transforms
def load_datasets():

    # Transforms for the image.
    transform = transforms.Compose([
                        transforms.Grayscale(3),
                        transforms.Resize(224,224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                ])

    # Define the image folder for each of the data set types
    trainset = torchvision.datasets.ImageFolder(
        root= "data/" + 'train',
        transform=transform
    )
    validset = torchvision.datasets.ImageFolder(
        root="data/" + 'val',
        transform=transform
    )
    testset = torchvision.datasets.ImageFolder(
        root="data/" + 'test',
        transform=transform
    )


    # Define indexes and get the subset random sample of each.
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
        
    return train_dataloader, valid_dataloader, test_dataloader