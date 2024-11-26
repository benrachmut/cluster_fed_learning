import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

# Define transformation (e.g., normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-100 dataset
train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Load the meta information for superclasses and subclasses
meta = train_set.meta  # Dictionary with superclass and subclass information
superclass_labels = meta['coarse_label_names']
subclass_labels = meta['fine_label_names']

# Specify the superclasses you want to filter
target_superclasses = ['flowers', 'vehicles 1']  # Replace with your desired superclasses
target_superclass_indices = [superclass_labels.index(sc) for sc in target_superclasses]

# Find the indices of the subclasses belonging to the target superclasses
target_subclass_indices = []
for idx, coarse_label in enumerate(train_set.coarse_labels):
    if coarse_label in target_superclass_indices:
        target_subclass_indices.append(idx)

# Create a subset of the dataset
filtered_train_set = Subset(train_set, target_subclass_indices)

# Create a DataLoader for the filtered dataset
train_loader = DataLoader(filtered_train_set, batch_size=32, shuffle=True)
