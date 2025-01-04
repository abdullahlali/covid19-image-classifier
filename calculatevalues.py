import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def calculate_mean_std(data_loader):
    mean = 0.
    std = 0.
    for images, _ in data_loader:
        batch_samples = images.size(0)  # number of images in the batch
        images = images.view(batch_samples, images.size(1), -1)  # flatten the images
        mean += images.mean(2).mean(0)  # mean for each channel
        std += images.std(2).std(0)  # std for each channel
    mean /= len(data_loader)
    std /= len(data_loader)
    return mean, std

# Example for your dataset
train_dataset = datasets.ImageFolder(root='/Users/abdullahali/Desktop/side_projects/covid19-image-classifier/COVID-19_Radiography_Dataset', transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

mean, std = calculate_mean_std(train_loader)
print(f'Mean: {mean}, Std: {std}')
