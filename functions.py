import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
from entities import *
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Subset



#### ----------------- IMPORT DATA ----------------- ####
def cut_data(data_set, size_use):
    if size_use > len(data_set):
        size_use = len(data_set)
    return torch.utils.data.Subset(data_set, range(int(size_use)))


def get_data_by_classification(data_set):
    class_labels = data_set.classes
    data_by_classification = defaultdict(list)
    for image, label in data_set:
        class_name = class_labels[label]
        data_by_classification[class_name].append(image)
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


def get_clients_and_server_data_portions(data_of_class, size_use):
    pass


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
    if data_set_selected == DataSet.CIFAR100:
        raise Exception("did not handle CIFAR100 yet")
    if data_set_selected == DataSet.CIFAR10:
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        data_by_classification_dict = get_data_by_classification(train_set)
        selected_classes_dict = get_selected_classes()
        clients_data_dict, server_data_dict,all_data = get_split_between_clients(data_by_classification_dict,selected_classes_dict)
        return selected_classes_dict, clients_data_dict, server_data_dict,all_data





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
        filtered_data_of_classes = filtered_data_of_classes[:test_set_size]

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

def create_clients(client_data_dict,server_data_dict,test_set):
    ans = []
    ids_list = []
    id_ = 0
    for class_, data_list in client_data_dict.items():
        for data_ in data_list:
            ids_list.append(id_)
            id_ = id_+1
            ans.append(Client(id_ =id_,client_data = data_,server_data=server_data_dict,test_data =test_set,class_ = class_ ))
    return ans,ids_list


def create_mean_df(clients, file_name):
    # List to hold the results for each iteration
    mean_results = []

    for t in range(iterations):
        # Gather test and train losses for the current iteration from all clients
        test_losses = []
        train_losses = []

        for c in clients:
            # Extract test losses for the current iteration
            test_loss_values = c.eval_test_df.loc[c.eval_test_df['Iteration'] == t, 'Test Loss'].values
            test_losses.extend(test_loss_values)  # Add the current client's test losses

            # Extract train losses for the current iteration
            train_loss_values = c.eval_test_df.loc[c.eval_test_df['Iteration'] == t, 'Train Loss'].values
            train_losses.extend(train_loss_values)  # Add the current client's train losses

        # Calculate the mean of the test and train losses, ignoring NaNs
        mean_test_loss = pd.Series(test_losses).mean()
        mean_train_loss = pd.Series(train_losses).mean()

        # Append a dictionary for this iteration to the list
        mean_results.append({
            'Iteration': t,
            'Average Test Loss': mean_test_loss,
            'Average Train Loss': mean_train_loss
        })

    # Convert the list of dictionaries into a DataFrame
    average_loss_df = pd.DataFrame(mean_results)

    # Save the DataFrame to a CSV file
    average_loss_df.to_csv(file_name + ".csv", index=False)

    return average_loss_df