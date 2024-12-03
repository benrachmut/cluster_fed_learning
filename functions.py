from collections import defaultdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
from entities import *
import matplotlib.pyplot as plt





#### ----------------- IMPORT DATA ----------------- ####
def cut_data(data_set, size_use):
    if size_use > len(data_set):
        size_use = len(data_set)
    return torch.utils.data.Subset(data_set, range(int(size_use)))



def get_data_by_classification(data_set):

    class_labels = data_set.classes
    data_by_classification = defaultdict(list)
    for single in data_set:
        image, label = single

        class_name = class_labels[label]
        data_by_classification[class_name].append(single)
    return data_by_classification


def get_list_of_superclass():
    sorted_keys = sorted(get_CIFAR10_superclass_dict().keys())
    num_superclass_local = num_superclass
    selected_keys = []
    for i in range(num_superclass_local):
        selected_keys.append(sorted_keys[i])
    return selected_keys


def get_dict_of_classes(list_of_superclass):
    ans ={}
    for superclass in list_of_superclass:
        ans[superclass]=[]
        list_of_classes = get_CIFAR10_superclass_dict()[superclass]
        sorted_list_of_classes = sorted(list_of_classes)
        num_class_per_super_class_local = num_classes_per_superclass
        for i in range(num_class_per_super_class_local):
            ans[superclass].append(sorted_list_of_classes[i])
    return ans

def get_selected_classes():
    if num_superclass>len(get_CIFAR10_superclass_dict()):
        raise Exception("num_superclass is larger then the subclasses in the data")
    list_of_superclass = get_list_of_superclass()
    dict_of_classes = get_dict_of_classes(list_of_superclass)
    return dict_of_classes


def get_split_between_clients(data_by_classification_dict, selected_classes):
    server_data_dict = {}
    clients_data_dict = {}
    all_train = []
    for superclass, classes_list in selected_classes.items():
        for current_class in classes_list:

            data_of_class = data_by_classification_dict[current_class]
            all_train = all_train+data_of_class
            train_set_size = len(data_of_class)
            size_use = int(percent_train_data_use * train_set_size)
            data_of_class = cut_data(data_of_class, size_use)
            client_data_per_class, server_data_per_class = split_clients_server_data(data_of_class)

            clients_data_dict[current_class] = client_data_per_class
            server_data_dict[current_class] = server_data_per_class
    return clients_data_dict,server_data_dict,all_train


def get_train_set():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(32, padding=4),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])

    # Load CIFAR-10 training dataset
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Load CIFAR-10 training dataset
    data_by_classification_dict = get_data_by_classification(train_set)
    selected_classes_dict = get_selected_classes()
    clients_data_dict, server_data_dict, all_data = get_split_between_clients(data_by_classification_dict,
                                                                              selected_classes_dict)
    return selected_classes_dict, clients_data_dict, server_data_dict, all_data


def get_test_set(test_set_size,selected_classes_dict):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    #print(len(test_set))
    # Get class indices for selected_classes
    selected_classes_list = []
    for classes_ in selected_classes_dict.values():
        for class_ in classes_:
            selected_classes_list.append(class_)

    data_by_classification = get_data_by_classification(test_set)
    filtered_data_of_classes = []
    for class_ in selected_classes_list:
        filtered_data_of_classes.extend(data_by_classification[class_])
        #print(len(data_by_classification[class_]))

    # Limit to the specified test_set_size
    if test_set_size < len(filtered_data_of_classes):
        filtered_data_of_classes = filtered_data_of_classes[:int(test_set_size)]

    return filtered_data_of_classes

def create_data():
    selected_classes_dict, clients_data_dict, server_data_dict, train_set = get_train_set()
    test_set_size = len(train_set) * percent_test_relative_to_train
    test_set = get_test_set(test_set_size,selected_classes_dict)

    print("train set size:",len(train_set))
    print("test set size:",len(test_set))

    return clients_data_dict, server_data_dict, test_set

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

    total_client_data_size = int((1-server_split_ratio) * len(train_set))
    server_data_size = len(train_set) - total_client_data_size
    client_data_size = total_client_data_size // identical_clients  # Each client gets an equal share
    split_sizes = [client_data_size] * identical_clients  # List of client dataset sizes
    split_sizes.append(server_data_size)  # Add the remaining data for the server
    splits = random_split(train_set, split_sizes)
    client_data_sets = splits[:-1]  # All client datasets
    server_data = splits[-1]

    return client_data_sets, server_data


#### ----------------- CREATE CLIENTS ----------------- ####

import torch
from torch.utils.data import DataLoader


def inspect_data(local_data):
    # Check the type of local data
    print("Data Type:", type(local_data))

    # Check if it's a DataLoader, and inspect the first batch
    if isinstance(local_data, DataLoader):
        print("DataLoader is being used. Let's check the first batch...")
        first_batch = next(iter(local_data))  # Fetch the first batch
        print("First Batch Type:", type(first_batch))
        print("First Batch Length:", len(first_batch))
        print("First Batch Content (input, target):", first_batch)

        # Check if the batch has two components (inputs and targets)
        if len(first_batch) == 2:
            inputs, targets = first_batch
            print("Input Shape:", inputs.shape if isinstance(inputs, torch.Tensor) else "Non-Tensor input")
            print("Target Shape:", targets.shape if isinstance(targets, torch.Tensor) else "Non-Tensor target")
        else:
            print("Warning: The batch does not contain 2 elements (inputs, targets). It contains:", len(first_batch),
                  "elements.")

    # Check if it's a list or dictionary (common for custom datasets)
    elif isinstance(local_data, (list, dict)):
        print("Local data is a list or dictionary.")
        if isinstance(local_data, dict):
            print("Local data is a dictionary. Keys:", list(local_data.keys()))
            for key in local_data:
                print(f"Sample data for class {key}:")
                print("  Type:", type(local_data[key]))
                print("  Length:", len(local_data[key]))
                print("  Example data:", local_data[key][:1])  # Print the first item in the class
        elif isinstance(local_data, list):
            print("Local data is a list. Length:", len(local_data))
            print("First 3 entries:", local_data[:3])

    # Check if it's a custom dataset object (like an instance of torchvision Dataset)
    elif hasattr(local_data, '__getitem__'):
        print("Local data is a custom dataset object (likely torchvision Dataset).")
        print("Example data entry (first item):", local_data[0])  # Example (image, label) pair
        print("Data type of first entry:", type(local_data[0]))
        print("Shape of input (image) in first entry:", local_data[0][0].shape)  # If it's an image tensor
        print("Label of first entry:", local_data[0][1])  # Assuming the label is the second element

    else:
        print("Unknown data type. Could not inspect further.")

    print("\n--- Data Inspection Completed ---")


def create_clients(client_data_dict,server_data_dict,test_set):
    ans = []
    ids_list = []
    id_ = 0
    for class_, data_list in client_data_dict.items():
        for data_ in data_list:
            #inspect_data(data_)
            ids_list.append(id_)

            id_ = id_+1
            ans.append(Client(id_ =id_,client_data = data_,global_data=server_data_dict,test_data =test_set,class_ = class_ ))
    return ans,ids_list

def create_csv(clients, server):
    sheets_names = []
    dfs = []
    eval_clients_df = []

    for c in clients:
        df = c.train_df
        dfs.append(df)
        sheets_names.append(c.id_ + "_train")

        df = c.eval_test_df
        dfs.append(df)
        sheets_names.append(c.id_ + "_eval")
        eval_clients_df.append(df)

    df = server.train_df
    dfs.append(df)
    sheets_names.append(server.id_ + "_train")

    df = server.eval_test_df
    dfs.append(df)
    sheets_names.append(server.id_ + "_eval")

    concatenated_df = pd.concat(eval_clients_df)

    # Ensure necessary columns exist
    required_columns = get_meta_data_text_keys()+[ 'Id', 'Iteration', 'Train Loss', 'Test Loss']



    # Initialize averaged_df as an empty DataFrame by default
    averaged_df = pd.DataFrame()

    # Check if all required columns are in the DataFrame
    if all(col in concatenated_df.columns for col in required_columns):
        # Group by the desired columns, calculating the mean for Train and Test Loss
        averaged_df = (
            concatenated_df.groupby(
                get_meta_data_text_keys()+[ 'Iteration'] )
                .agg({
                'Train Loss': 'mean',
                'Test Loss': 'mean'
            })
                .reset_index()
        )

    dfs.append(averaged_df)
    sheets_names.append("avg_eval")

    # Create a writer object and save each dataframe to a different sheet
    with pd.ExcelWriter(file_name() + ".xlsx", engine='xlsxwriter') as writer:
        for df, sheet in zip(dfs, sheets_names):
            df.to_excel(writer, sheet_name=sheet, index=False)

