import torch
import torch.nn as nn
import torchvision.models as models
from se_vgg16_fire import SEVGG16

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = SEVGG16(num_classes=2).to(device)

# Load pretrained weights from VGG16
pretrained_vgg16 = models.vgg16(pretrained=True)
pretrained_dict = pretrained_vgg16.state_dict()
model_dict = model.state_dict()

# Update model dictionary with pretrained weights where possible
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Save the pretrained model
torch.save(model.state_dict(), "pretrained_se_vgg16.pth")

print("Pretrained SE-VGG16 model saved successfully!")
