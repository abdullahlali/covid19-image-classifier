import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split


save_dir = '/Users/abdullahali/Desktop/side_projects/covid19-image-classifier'
data_dir = '/Users/abdullahali/Desktop/side_projects/covid19-image-classifier/COVID-19_Radiography_Dataset'


# Use the calculated mean and std
mean = [0.5095, 0.5095, 0.5095]  # RGB channels mean
std = [0.0432, 0.0432, 0.0432]  # RGB channels std



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}



dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])

# Split the dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Save the datasets
torch.save(train_dataset, os.path.join(save_dir, 'train_dataset.pth'))
torch.save(test_dataset, os.path.join(save_dir, 'test_dataset.pth'))

print("Datasets split and saved as binary files!")