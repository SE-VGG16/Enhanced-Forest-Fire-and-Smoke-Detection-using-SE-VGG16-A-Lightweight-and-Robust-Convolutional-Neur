import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset_preprocessing import val_loader
from se_vgg16_fire import SEVGG16

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = SEVGG16(num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pth"))  # Ensure you have saved the best model
model.eval()

# Initialize lists to store predictions and ground truth labels
y_true = []
y_pred = []

# Evaluation loop
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
