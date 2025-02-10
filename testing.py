import torch
import torchvision.transforms as transforms
from PIL import Image
from se_vgg16_fire import SEVGG16

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = SEVGG16(num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pth"))  # Ensure you have saved the best model
model.eval()

# Define preprocessing transformations
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Function to make predictions
def predict(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    return "Forest Fire" if predicted_class.item() == 1 else "Normal"

# Test an image
image_path = "test_image.jpg"  # Replace with actual test image path
result = predict(image_path)
print(f"Prediction: {result}")
