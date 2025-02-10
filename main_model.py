import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        se = self.global_avg_pool(x).view(batch, channels)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(batch, channels, 1, 1)
        return x * se

class SEVGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(SEVGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        
        # Insert SE blocks after each convolutional block
        self.features = nn.Sequential(
            *features[:4], SEBlock(64),
            *features[4:9], SEBlock(128),
            *features[9:16], SEBlock(256),
            *features[16:23], SEBlock(512),
            *features[23:30], SEBlock(512)
        )
        
        self.avgpool = vgg16.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Model initialization
model = SEVGG16(num_classes=2)
print(model)
