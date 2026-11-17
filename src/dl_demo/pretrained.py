import torch.nn as nn
from torchvision import models


class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.model(x)
