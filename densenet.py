import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
