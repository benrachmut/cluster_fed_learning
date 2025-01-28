import copy

import pandas as pd
from scipy.stats import ansari
from torch.utils.data import TensorDataset, random_split,Subset
from torchvision.transforms import transforms

from entities import *
from collections import defaultdict
import random as rnd



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


def create_torch_mix_set(data_to_mix_list):
    combined_indices = []
    dataset = None
    all_samples = []  # To store (index, sample) pairs for better control of shuffling

    for subset in data_to_mix_list:
        if dataset is None:
            dataset = subset.dataset  # Get the dataset reference from the first subset

        # Pair each index with its data for better mixing
        all_samples.extend((idx, dataset[idx]) for idx in subset.indices)

    # Step 2: Shuffle the list of samples
    generator = torch.Generator().manual_seed(42)
    shuffled_samples = torch.randperm(len(all_samples), generator=generator).tolist()

    # Step 3: Extract only the indices from the shuffled (index, sample) pairs
    shuffled_indices = [all_samples[i][0] for i in shuffled_samples]

    # Step 4: Create a new Subset using the shuffled indices
    mixed_subset = Subset(dataset, shuffled_indices)
    return mixed_subset



def split_list(data, num_splits, seed=None):

    if seed is not None:
        rnd.seed(17)

    # Shuffle the data
    rnd.shuffle(data)

    # Calculate sizes for splits
    total_size = len(data)
    split_size = total_size // num_splits
    split_sizes = [split_size] * num_splits

    # Distribute the remainder among the splits
    for i in range(total_size % num_splits):
        split_sizes[i] += 1

    # Split the data
    splits = []
    start = 0
    for size in split_sizes:
        splits.append(data[start:start + size])
        start += size

    return splits
def get_mix_torch(data_to_mix_list):
    all_mix_tuples = []
    for l in data_to_mix_list:
        for image in l:
            all_mix_tuples.append(image)
    rnd.seed(17)
    rnd.shuffle(all_mix_tuples)
    return split_list(all_mix_tuples, experiment_config.num_clients, 1357)

def get_data_to_mix_and_data_to_leave(classes_data_dict):
    data_to_mix_list = []
    clients_original_data_dict = {}

    for class_name, single_tensor_set in classes_data_dict.items():
        size_to_mix = int(len(single_tensor_set) * experiment_config.mix_percentage)
        size_to_leave = int(len(single_tensor_set) * (1 - experiment_config.mix_percentage))

        torch.Generator().manual_seed(777)
        splits = random_split(single_tensor_set, [size_to_mix, size_to_leave])
        data_to_mix = transform_to_TensorDataset(splits[0])
        data_to_mix_list.append(data_to_mix)
        data_to_leave = transform_to_TensorDataset(splits[1])
        clients_original_data_dict[class_name] = data_to_leave
    return data_to_mix_list,clients_original_data_dict


def get_images_per_group_dict(group,target_original_data_dict):
    images_per_group_dict = {}
    for target in group:
        t_data_set = target_original_data_dict[target]
        i = 0
        for image in t_data_set:
            if i not in images_per_group_dict:
                images_per_group_dict[i] = []
            images_per_group_dict[i].append(image)
            if i == len(group) - 1:
                i = 0
            else:
                i = i + 1
    ans = []
    for v in images_per_group_dict.values():
        ans.append(transform_to_TensorDataset(v))
    return ans

def get_mix_tensor_list(target_original_data_dict):

    identical_clients = experiment_config.identical_clients
    # Step 1: Get all the keys


    rnd.seed(19)
    keys = list(target_original_data_dict.keys())
    rnd.shuffle(keys)
    groups = [keys[i:i + identical_clients] for i in range(0, len(keys), identical_clients)]

    list_of_torches_clients = []
    for group in groups:
        list_of_group = get_images_per_group_dict(group,target_original_data_dict)
        list_of_torches_clients.extend(list_of_group)

    return list_of_torches_clients

def get_match_mix_clients(mix_tensor_list,clients_tensor_list):
    ans = []
    for i in range(len(mix_tensor_list)):
        data_per_client_list = []
        first_list = mix_tensor_list[i]
        second_list = clients_tensor_list[i]
        for image in first_list:
            data_per_client_list.append(image)
        for image in second_list:
            data_per_client_list.append(image)

        rnd.seed(658+17*(i+1))
        rnd.shuffle(data_per_client_list)
        ans.append(transform_to_TensorDataset(data_per_client_list))
    return ans


def get_split_train_client_data(classes_data_dict):
    data_to_mix_list,target_original_data_dict = get_data_to_mix_and_data_to_leave(classes_data_dict)
    mix_tensor_list = get_mix_torch(data_to_mix_list)
    clients_tensor_list = get_mix_tensor_list(target_original_data_dict)
    if len(mix_tensor_list)!=len(clients_tensor_list):raise Exception("lists need to be same length")
    match_mix_clients = get_match_mix_clients(mix_tensor_list,clients_tensor_list)
    ans = {}
    for i in range(len(match_mix_clients)):ans[i]=match_mix_clients[i]



    return ans







    #for class_name, image_groups in clients_data_dict.items():
    #    reduced_image_groups = []
    #    clients_original_image_group = []
    #    for group in image_groups:
    #        # Calculate the number of items to include based on the percentage
    #        num_images_to_include = int(len(group) * experiment_config.mix_percentage)
    #        images_left = int(len(group) * (1-experiment_config.mix_percentage))
    #        split_sizes = [images_left,num_images_to_include]
    #        splits = random_split(group, split_sizes)
    #        clients_original_image_group.append(splits[0])
    #        for image_ in splits[1]:
    #            reduced_image_groups.append(image_)

     #   clients_original_data_dict[class_name] = clients_original_image_group
     #   data_to_mix.extend(reduced_image_groups)
     #   rnd.seed(17)
    #    rnd.shuffle(data_to_mix)
    #return clients_original_data_dict,data_to_mix



def complete_client_data(clients_data_dict, data_to_mix,data_size_per_client):
    ans={}
    counter = 0
    for class_name, client_lists in clients_data_dict.items():
        ans[class_name] = []
        for client_list in client_lists:
            counter = +1
            new_subset = []

            new_data_to_mix = []


            for image in client_list:
                new_subset.append(image)

            for image in data_to_mix:
                if len(new_subset) < data_size_per_client and image[1] != class_name:
                    new_subset.append(image)
                else:
                    new_data_to_mix.append(image)
            data_to_mix = new_data_to_mix

            rnd.seed(counter*17)
            rnd.shuffle(new_subset)
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


def transform_to_TensorDataset(data_):
    images = [item[0] for item in data_]  # Extract the image tensors (index 0 of each tuple)
    targets = [item[1] for item in data_]

    # Step 2: Convert the lists of images and targets into tensors (if not already)
    images_tensor = torch.stack(images)  # Stack the image tensors into a single tensor
    targets_tensor = torch.tensor(targets)  # Convert the targets to a tensor


    # Step 3: Create a TensorDataset from the images and targets
    return TensorDataset(images_tensor, targets_tensor)



def create_identical_groups(clients_data_dict):
    identical_clients = experiment_config.identical_clients

    # Step 1: Get all the keys
    keys = list(clients_data_dict.keys())

    # Step 2: Create groups of the desired size
    groups = [keys[i:i + identical_clients] for i in range(0, len(keys), identical_clients)]
    for group in groups:
        sets_of_data = {}
        for client_id in group:
            sets_of_data[client_id] = clients_data_dict[client_id]

            print(type(sets_of_data[client_id][0]))
            print()



            #for image in client_list:
            #    new_subset.append(image)

            #for image in data_to_mix:
            #    if len(new_subset) < data_size_per_client and image[1] != class_name:
            #        new_subset.append(image)
            #    else:
            #        new_data_to_mix.append(image)
            #data_to_mix = new_data_to_mix

            #rnd.seed(counter * 17)
            #rnd.shuffle(new_subset)
            #new_td = transform_to_TensorDataset(new_subset)
    print()
def cut_data_for_partial_use_of_data(class_target,data_by_classification_dict):
    data_of_class = data_by_classification_dict[class_target]
    train_set_size = len(data_of_class)
    size_use = int(experiment_config.percent_train_data_use * train_set_size)
    return cut_data(data_of_class, size_use)

def get_server_data(server_data_dict):
    server_data = create_server_data(server_data_dict)
    rnd.seed(42)
    rnd.shuffle(server_data)
    return transform_to_TensorDataset(server_data)
def split_clients_server_data_Non_IID(data_by_classification_dict, selected_classes):
    server_data_dict = {}
    classes_data_dict = {}
    for class_target in selected_classes:
        data_of_class = cut_data_for_partial_use_of_data(class_target,data_by_classification_dict)
        client_data_per_class, server_data_per_class = split_clients_server_data(data_of_class)
        classes_data_dict[class_target] = client_data_per_class
        server_data_dict[class_target] = server_data_per_class
    server_data = get_server_data(server_data_dict)

    clients_data_dict = get_split_train_client_data(classes_data_dict)
    return clients_data_dict,server_data




def split_clients_server_data_IID(train_set,server_split_ratio):
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
    client_data_size = total_client_data_size // experiment_config.num_clients  # Each client gets an equal share
    split_sizes = [client_data_size] * experiment_config.num_clients  # List of client dataset sizes
    split_sizes.append(server_data_size)  # Add the remaining data for the server
    seed = 42
    torch.manual_seed(seed)
    splits = random_split(train_set, split_sizes)
    client_data_sets = splits[:-1]  # All client datasets
    server_data = splits[-1]

    clients_data_dict = change_format_of_clients_data_dict(client_data_sets)
    server_data = transform_to_TensorDataset(server_data)

    return clients_data_dict, server_data


def change_format_of_clients_data_dict(client_data_sets):
    clients_data_dict = {}
    counter = 0

    for single_set in client_data_sets:
        image_list = []
        for set_ in single_set:
            image_list.append(set_)
        clients_data_dict[counter] = transform_to_TensorDataset(image_list)
        counter += 1
    return clients_data_dict


def get_data_set(data_type,is_train ):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(32, padding=4),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])


    train_set = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)
    data_by_classification_dict = get_data_by_classification(train_set)
    selected_classes_list = sorted(data_by_classification_dict.keys())[:experiment_config.num_classes]

    if data_type == DataType.IID:
        clients_data_dict, server_data = split_clients_server_data_IID(train_set, experiment_config.server_split_ratio)

    if data_type == DataType.NonIID:
        clients_data_dict, server_data = split_clients_server_data_Non_IID(data_by_classification_dict, selected_classes_list)

    return selected_classes_list, clients_data_dict, server_data





def get_test_set(test_set_size,selected_classes_list):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    data_by_classification = get_data_by_classification(test_set)
    filtered_data_of_classes = []
    for class_ in selected_classes_list:
        filtered_data_of_classes.extend(data_by_classification[class_])
        #print(len(data_by_classification[class_]))

    # Limit to the specified test_set_size
    #if test_set_size < len(filtered_data_of_classes):
    #    filtered_data_of_classes = filtered_data_of_classes[:test_set_size]

    return filtered_data_of_classes


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


def create_data(data_type):
    selected_classes_list, clients_train_data_dict, server_train_data = get_data_set(data_type, is_train = True)
    #train_set_size = get_train_set_size(clients_data_dict, server_data)
    #test_set_size = train_set_size * experiment_config.percent_test_relative_to_train
    selected_test_classes_list, clients_test_data_dict, server_test_data = get_data_set(data_type, is_train = False)

    #test_set = get_test_set(test_set_size,selected_classes_list)
    # TODO get test data by what it if familar with + what it is not familiar with.
    #test_set =transform_to_TensorDataset(test_set)
   # print_data_for_debug(clients_data_dict,server_data, test_set)


    return clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data

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

    train_set_images = []
    for image in train_set: train_set_images.append(image)

    total_client_data_size = int((1-experiment_config.server_split_ratio) * len(train_set))
    server_data_size = len(train_set) - total_client_data_size
    client_data_size = total_client_data_size #// experiment_config.identical_clients  # Each client gets an equal share
    split_sizes = [client_data_size,server_data_size] #* experiment_config.identical_clients  # List of client dataset sizes

    torch.Generator().manual_seed(999)
    splits = random_split(train_set_images, split_sizes)
    clients_data = transform_to_TensorDataset(splits[0])  # All client datasets
    server_data = transform_to_TensorDataset(splits[1])

    return clients_data, server_data



#### ----------------- CREATE CLIENTS ----------------- ####

def create_clients(client_data_dict,server_data,test_set):
    ans = []
    ids_list = []


    for id_, data_ in client_data_dict.items():
        ids_list.append(id_)
        ans.append(Client(id_ =id_,client_data = data_,global_data=server_data,test_data =test_set[id_]))


    return ans,ids_list


def create_mean_df(clients, file_name):
    # List to hold the results for each iteration
    mean_results = []

    for t in range(experiment_config.iterations):
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