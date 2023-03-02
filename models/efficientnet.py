import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.num_classes = num_classes
        self.fc = nn.Linear(self.model._fc.in_features, self.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x
