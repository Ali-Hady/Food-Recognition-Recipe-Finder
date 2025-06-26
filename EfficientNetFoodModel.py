import torch
from torch import nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class FoodClassifierEfficientNetB3Based(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b3(weights=weights)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)