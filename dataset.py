import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision import transforms, datasets
import numpy as np
import torchvision.models as models

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# Get the transforms
def load_datasets(batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder('data', transform=transform)
    num_samples = len(dataset)
    train_size = int(num_samples * 0.8)
    val_size = (num_samples - train_size) // 2
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader