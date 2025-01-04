import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def load_model(model_path, num_classes):
    """Load the model with the specified number of classes and state dictionary."""
    # Define the model architecture (e.g., ResNet18)
    model = models.resnet18(pretrained=False)  # Change this to match your model
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the output layer
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    
    return model

# Use the calculated mean and std
mean = [0.5095, 0.5095, 0.5095]  # RGB channels mean
std = [0.0432, 0.0432, 0.0432]  # RGB channels std

def preprocess_image(image_path):
    """Preprocess the image to match the model input."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(image_path, model, class_labels):
    """Predict the class of an image."""
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
    return class_labels[predicted_class.item()]

def main():
    base_dir = '/Users/abdullahali/Desktop/side_projects/covid19-image-classifier'
    model_path = os.path.join(base_dir, 'covid19-model.pth')
    num_classes = 4  # Adjust based on your dataset
    model = load_model(model_path, num_classes)

    class_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']  # Update based on your dataset

    images = ['covid.png', 'lungopacity.png', 'normal.png', 'normal2.png', 'pneumonia.png', 'pneumonia2.png']

    for image_path in images:
        predicted_label = predict(image_path, model, class_labels)
        print(f'\nInput Image: {image_path}')
        print(f'Predicted class: {predicted_label}')


if __name__ == '__main__':
    main()