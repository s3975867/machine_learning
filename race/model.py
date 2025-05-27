from torch import nn
from torchvision.models import resnet18

class AgeClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(AgeClassifier, self).__init__()
        self.base = resnet18(pretrained=pretrained)
        
        num_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.base(x)