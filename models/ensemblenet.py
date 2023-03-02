import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class EnsembleNet(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleNet, self).__init__()
        
        # Define the models to ensemble
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
        self.inceptionv3 = models.inception_v3(pretrained=True)
        self.resnet50 = models.resnet50(pretrained=True)
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Freeze the parameters of the pre-trained models
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.inceptionv3.parameters():
            param.requires_grad = False
        for param in self.resnet50.parameters():
            param.requires_grad = False
        for param in self.vgg16.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer of each model with a new one
        num_ftrs = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = self.inceptionv3.fc.in_features
        self.inceptionv3.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
        # Define the aggregation layer
        self.aggregation_layer = nn.Linear(4*num_classes, num_classes)

    def forward(self, x):
        # Forward pass through each model
        out_efficientnet = self.efficientnet(x)
        out_inceptionv3 = self.inceptionv3(x)
        out_resnet50 = self.resnet50(x)
        out_vgg16 = self.vgg16(x)
        
        # Concatenate the outputs
        out = torch.cat((out_efficientnet, out_inceptionv3, out_resnet50, out_vgg16), dim=1)
        
        # Pass through the aggregation layer
        out = self.aggregation_layer(out)
        
        return out