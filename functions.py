import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
from entities import *



#### ----------------- IMPORT DATA ----------------- ####
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
    size_use = int(percent_train_data_use * train_set_size)

    ans = cut_data(ans, size_use)
    return ans


def get_test_set(test_set_size):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])
    ans = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    ans = cut_data(ans, test_set_size)
    return ans


def create_data():
    train_set = get_train_set()
    test_set_size = len(train_set) * percent_test_relative_to_train
    test_set = get_test_set(test_set_size)
    return train_set, test_set

#### ----------------- SPLIT DATA BETWEEN SERVER AND CLIENTS ----------------- ####

def split_clients_server_data(train_set):
    """
        Splits the input training dataset into subsets for multiple clients and the server.

        Args:
        - train_set: The full training dataset to be split.

        Returns:
        - client_data_sets: A list of datasets for each client.
        - server_data: A dataset for the server.

        The function dynamically allocates the training data based on the number of clients and a specified split ratio for the server.
        """

    total_client_data_size = int(client_split_ratio * len(train_set))
    server_data_size = len(train_set) - total_client_data_size
    client_data_size = total_client_data_size // num_clients  # Each client gets an equal share
    split_sizes = [client_data_size] * num_clients  # List of client dataset sizes
    split_sizes.append(server_data_size)  # Add the remaining data for the server
    splits = random_split(train_set, split_sizes)
    client_data_sets = splits[:-1]  # All client datasets
    server_data = splits[-1]

    return client_data_sets, server_data



#### ----------------- CREATE CLIENTS ----------------- ####

def create_clients(client_data_sets,server_data):
    ans = []
    for i in range(len(client_data_sets)):
        client_data = client_data_sets[i]
        ans.append(Client(id_ =i,client_data = client_data,server_data=server_data))
    return ans