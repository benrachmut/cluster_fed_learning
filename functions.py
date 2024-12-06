import copy
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, random_split
from entities import *
from collections import defaultdict
import random as rnd
from config import *



#### ----------------- IMPORT DATA ----------------- ####
def cut_data(data_set, size_use):
    if size_use > len(data_set):
        size_use = len(data_set)
    return torch.utils.data.Subset(data_set, range(int(size_use)))


def get_data_by_classification(data_set):
    #class_labels = data_set.classes
    data_by_classification = defaultdict(list)
    for image in data_set:
    #for image, label in data_set:
        data_by_classification[image[1]].append(image)
    return data_by_classification









def get_clients_and_server_data_portions(data_of_class, size_use):
    pass

def get_split_train_client_data(clients_data_dict):
    """
    Splits the train data by a given percentage for each list in the dictionary.

    Args:
        clients_data_dict (dict): Dictionary where keys are class names and values are lists of lists of images.
        percentage (float): Percentage (0-100) of data to keep from each list.

    Returns:
        dict: New dictionary with the same structure but containing the reduced data.
    """
    reduced_data_dict = {}
    clients_original_data_dict = {}
    for class_name, image_groups in clients_data_dict.items():
        reduced_image_groups = []
        clients_original_image_group = []
        for group in image_groups:
            # Calculate the number of items to include based on the percentage
            num_images_to_include = int(len(group) * mix_percentage)
            images_left = int(len(group) * (1-mix_percentage))
            split_sizes = [images_left,num_images_to_include]
            splits = random_split(group, split_sizes)
            clients_original_image_group.append(splits[0])
            reduced_image_groups.append(splits[1])

        clients_original_data_dict[class_name] = clients_original_image_group
        reduced_data_dict[class_name] = reduced_image_groups

    return clients_original_data_dict,reduced_data_dict



def complete_client_data(clients_data_dict, data_to_mix):
    """
    Completes each client's data in clients_data_dict to the target size
    using data from split_train_client_data, ensuring the additional data
    comes from different classes.

    Args:
        clients_data_dict (dict): Original client data, with keys as class names and values as lists of image lists.
        split_train_client_data (dict): Additional data to use for completion.
        target_size (int): Desired size for each client's list.

    Returns:
        dict: Updated clients_data_dict with completed data.
    """
    ans={}
    for class_name, client_lists in clients_data_dict.items():
        ans[class_name] = []
        for client_list in client_lists:

            other_classes = list(data_to_mix.keys())
            if class_name in other_classes:
                other_classes.remove(class_name)
            other_class_selected = rnd.choice(other_classes)

            other_data_selected = data_to_mix[other_class_selected].pop(0)
            if len(data_to_mix[other_class_selected])==0:
                del data_to_mix[other_class_selected]
            new_subset = []
            for image in client_list:new_subset.append(image)
            for image in other_data_selected: new_subset.append(image)

            new_td = transform_to_TensorDataset(new_subset)
            ans[class_name].append(new_td)

            #rnd.shuffle()

    return ans


def check_data_targets(data_,name_of_data):
    targets = torch.tensor([target for _, target in data_])
    unique_labels = torch.unique(targets)

    print("unique targets for",name_of_data,str(unique_labels))



def create_server_data(server_data_dict):
    all_subsets = []
    all_images = []
    for v  in server_data_dict.values():
        all_subsets.append(v)
        for image in v:
            all_images.append(image)

    #check_server_data_targets(ans)

    return all_images



def get_split_between_entities(data_by_classification_dict, selected_classes):
    server_data_dict = {}
    clients_data_dict = {}
    for class_target in selected_classes:
        data_of_class = data_by_classification_dict[class_target]
        train_set_size = len(data_of_class)
        size_use = int(percent_train_data_use * train_set_size)
        data_of_class = cut_data(data_of_class, size_use)
        client_data_per_class, server_data_per_class = split_clients_server_data(data_of_class)
        clients_data_dict[class_target] = client_data_per_class
        server_data_dict[class_target] = server_data_per_class
    server_data = create_server_data(server_data_dict)
    server_data = transform_to_TensorDataset(server_data)
    clients_data_dict, data_to_mix = get_split_train_client_data(clients_data_dict)
    clients_data_dict = complete_client_data(clients_data_dict, data_to_mix)




    return clients_data_dict,server_data




def get_train_set():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(32, padding=4),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])


    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_by_classification_dict = get_data_by_classification(train_set)
    selected_classes_list = sorted(data_by_classification_dict.keys())[:num_classes]
    clients_data_dict, server_data = get_split_between_entities(data_by_classification_dict, selected_classes_list)
    return selected_classes_list, clients_data_dict, server_data





def get_test_set(test_set_size,selected_classes_list):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    data_by_classification = get_data_by_classification(test_set)
    test_data_list = []
    for class_ in selected_classes_list:
        test_data_list.extend(data_by_classification[class_])
        #print(len(data_by_classification[class_]))

    test_data_dict = {}
    for class_ in selected_classes_list:
        test_data_dict[class_]=data_by_classification[class_]

    # Limit to the specified test_set_size
    

    test_data_list =transform_to_TensorDataset(test_data_list)

    return test_data_list,test_data_dict


def get_train_set_size(clients_data_dict, server_data):
    ans = 0
    for class_, datas in clients_data_dict.items():
        for data_ in datas:
           ans = ans + len(data_)
    ans = ans+len(server_data)
    return ans


def print_data_for_debug(clients_data_dict,server_data, test_set):
    for class_name, datas in clients_data_dict.items():
        sizes_ = []
        counter = 0
        for data_ in datas:
            sizes_.append(len(data_))
            check_data_targets(data_,"client"+str(counter))
            counter += 1
        print("avg train set size for", class_name, "is:", str(sum(sizes_) / len(sizes_)), "; total data in class",
              sum(sizes_))
    print("server data size:", len(server_data))
    check_data_targets(server_data, "server" + str(counter))

    print("test set size:", len(test_set))
    check_data_targets(test_set, "test" + str(counter))


def create_data():
    selected_classes_list, clients_data_dict, server_data = get_train_set()
    train_set_size = get_train_set_size(clients_data_dict, server_data)
    test_set_size = train_set_size * percent_test_relative_to_train
    test_set,test_data_dict = get_test_set(test_set_size,selected_classes_list)
    # TODO get test data by what it if familar with + what it is not familiar with.
    print_data_for_debug(clients_data_dict,server_data, test_set)


    return clients_data_dict, server_data, test_set,test_data_dict

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

def create_clients(client_data_dict,server_data_dict,test_set,test_data_dict):
    ans = []
    ids_list = []
    id_ = 0
    for class_, data_list in client_data_dict.items():
        for data_ in data_list:
            ids_list.append(id_)
            ans.append(Client(id_ =id_,client_data = data_,global_data=server_data_dict,test_data =test_set,class_ = class_,test_data_dict = test_data_dict ))
            id_ = id_+1

    return ans,ids_list







def create_record_data(clients, server):
    loss_measures = {}
    loss_measures_class_yes = {}
    loss_measures_class_no = {}

    accuracy_measures = {}
    accuracy_measures_class_yes = {}
    accuracy_measures_class_no = {}

    for client in clients:
        loss_measures["Client"+str(client.id_)]=client.loss_measures
        loss_measures_class_yes["Client"+str(client.id_)] = client.loss_measures_class_yes
        loss_measures_class_no["Client"+str(client.id_)] = client.loss_measures_class_no

        accuracy_measures["Client"+str(client.id_)]=client.accuracy_measures
        accuracy_measures_class_yes["Client" + str(client.id_)] = client.accuracy_measures_class_yes
        accuracy_measures_class_no["Client" + str(client.id_)] = client.accuracy_measures_class_no

    loss_measures[server.id_] = server.loss_measures
    accuracy_measures[server.id_] = server.accuracy_measures
    return RecordData(loss_measures,loss_measures_class_yes,loss_measures_class_no,accuracy_measures,accuracy_measures_class_yes,accuracy_measures_class_no)


def create_pickle(clients, server):
    rd = create_record_data(clients, server)
    pik_name = rd.summary
    pickle_file_path = pik_name+".pkl"

    with open(pickle_file_path, "wb") as file:
        pickle.dump(rd, file)