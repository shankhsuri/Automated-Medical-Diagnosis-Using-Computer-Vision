import torch.nn as nn
import torchvision.models as models

class Inceptionv3(nn.Module):
    def __init__(self, num_classes=2):
        super(Inceptionv3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        num_features = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_features, num_classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x