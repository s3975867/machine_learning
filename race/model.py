from torch import nn
from torchvision.models import resnet18

class AgeClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(AgeClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((224, 224))
        )
        self.backbone = resnet18(pretrained=pretrained)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
        )
        
    def forward(self, x):
        x = self.cnn(x)
        return self.backbone(x)