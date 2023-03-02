import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def show_images(dataloader, classes, num_images=25):
    """
    Show a grid of images from a given data loader.
    """
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    grid = make_grid(images, nrow=5)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title([classes[label] for label in labels])
    plt.axis('off')
    plt.show()

from dataset import load_datasets
from visualization import show_images

train_loader, valid_loader, test_loader = load_datasets()

classes = ['NORMAL', 'PNEUMONIA']

# Show a grid of images from the training set
show_images(train_loader, classes)    