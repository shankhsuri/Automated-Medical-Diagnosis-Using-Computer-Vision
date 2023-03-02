import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import load_datasets
from models.efficientnet import EfficientNet
from models.inceptionv3 import InceptionV3
from models.resnet50 import ResNet50
from models.vgg16 import VGG16
from models.ensemblenet import EnsembleNet
from visualization import show_images


# Define the device to use for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
learning_rate = 0.001
num_epochs = 10

# Load the data sets
train_loader, valid_loader, test_loader = load_datasets()

# Define the models
efficientnet = EfficientNet().to(device)
inceptionv3 = InceptionV3().to(device)
resnet50 = ResNet50().to(device)
vgg16 = VGG16().to(device)
ensemblenet = EnsembleNet().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ensemblenet.parameters(), lr=learning_rate)

# Train the models
models = [efficientnet, inceptionv3, resnet50, vgg16, ensemblenet]
for model in models:
    print(f"Training {model.__class__.__name__}")
    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader)}")

# Test the models
for model in models:
    print(f"Evaluating {model.__class__.__name__}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

# Show a grid of images from the training set
classes = ['NORMAL', 'PNEUMONIA']
show_images(train_loader, classes)