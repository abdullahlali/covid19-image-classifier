import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter
import numpy as np
from tqdm import tqdm

base_dir = '/Users/abdullahali/Desktop/side_projects/covid19-image-classifier'
data_dir = '/Users/abdullahali/Desktop/side_projects/covid19-image-classifier/COVID-19_Radiography_Dataset'

def main():
    # Load datasets
    train_dataset = torch.load(os.path.join(base_dir, 'train_dataset.pth'))
    test_dataset = torch.load(os.path.join(base_dir, 'test_dataset.pth'))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Define model
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    num_ftrs = model.fc.in_features
    dataset = datasets.ImageFolder(data_dir)
    model.fc = nn.Linear(num_ftrs, len(dataset.classes))

    # Assuming `dataset` is your ImageFolder dataset
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    total_samples = sum(class_counts.values())
    weights = [total_samples / count for count in class_counts.values()]
    weights = torch.tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model = model.to(device)
    print(f"Using device: {device}")

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm to add a progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Update tqdm description
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))  # Show the average loss so far
        
        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # Evaluate the model and calculate accuracy
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')


    model_path = os.path.join(base_dir, 'covid19-model.pth')
    torch.save(model.state_dict(), model_path)
    print("Model saved!")


if __name__ == '__main__':  
    main()
