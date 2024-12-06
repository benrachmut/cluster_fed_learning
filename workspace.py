import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from xlsxwriter import *

def cut_data(data_set, size_use):
    if size_use > len(data_set):
        size_use = len(data_set)
    return torch.utils.data.Subset(data_set, range(int(size_use)))

def get_train_set():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(32, padding=4),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])

    # Load CIFAR-10 training dataset
    ans = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_set_size = len(ans)
    size_use = int(1 * train_set_size)

    ans = cut_data(ans, size_use)
    return ans



def get_test_set(test_set_size):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])
    ans = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    if len(ans)<test_set_size:
        test_set_size = len(ans)
    ans = cut_data(ans, test_set_size)
    return ans



def split_clients_server_data(train_set,server_split_ratio):
    """
        Splits the input training dataset into subsets for multiple clients and the server.

        Args:
        - train_set: The full training dataset to be split.

        Returns:
        - client_data_sets: A list of datasets for each client.
        - server_data: A dataset for the server.

        The function dynamically allocates the training data based on the number of clients and a specified split ratio for the server.
        """

    total_client_data_size = int((1-server_split_ratio) * len(train_set))
    server_data_size = len(train_set) - total_client_data_size
    client_data_size = total_client_data_size // 4  # Each client gets an equal share
    split_sizes = [client_data_size] * 4  # List of client dataset sizes
    split_sizes.append(server_data_size)  # Add the remaining data for the server
    splits = random_split(train_set, split_sizes)
    client_data_sets = splits[:-1]  # All client datasets
    server_data = splits[-1]

    return client_data_sets, server_data



def create_data():
    train_set = get_train_set()
    test_set_size = len(train_set) * 1
    test_set = get_test_set(test_set_size)

    print("train set size:",len(train_set))
    print("test set size:",len(test_set))

    return train_set, test_set


if __name__ == '__main__':
    torch.manual_seed(1)  # Set seed for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)  # Set seed for all GPUs

    print("server_split_ratio:",0.2)
    train_set, test_set = create_data()
    client_data_sets,server_data = split_clients_server_data(train_set,0.2)
    print()

