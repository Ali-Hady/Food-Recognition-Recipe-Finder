import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class FoodClassifierResNetBased(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.base_model = resnet50(weights=weights)
        #self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)    