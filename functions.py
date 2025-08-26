import os
import shutil
import tarfile
import urllib
import zipfile
from urllib.request import urlretrieve

import pandas as pd

from torch.utils.data import TensorDataset, random_split, Subset, Dataset
from torchvision.datasets import ImageFolder, EMNIST
from torchvision.transforms import transforms

from entities import *
from collections import defaultdict
import random as rnd

from typing import Dict, List, Tuple, Optional


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

def get_data_by_classification_memory_efficient(data_set):
    """
    Memory-efficient version that stores indices instead of actual data.
    This prevents loading all images into memory at once.
    """
    data_by_classification = defaultdict(list)
    
    # Store indices instead of actual data
    for idx in range(len(data_set)):
        sample = data_set[idx]
        if isinstance(sample, tuple) and len(sample) == 2:
            _, label = sample
        else:
            label = sample[1]
        
        data_by_classification[label].append(idx)
    
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
    generator = torch.Generator().manual_seed(42*experiment_config.seed_num)
    shuffled_samples = torch.randperm(len(all_samples), generator=generator).tolist()

    # Step 3: Extract only the indices from the shuffled (index, sample) pairs
    shuffled_indices = [all_samples[i][0] for i in shuffled_samples]

    # Step 4: Create a new Subset using the shuffled indices
    mixed_subset = Subset(dataset, shuffled_indices)
    return mixed_subset



def split_list(data, num_splits, seed=None):

    if seed is not None:
        rnd.seed(experiment_config.seed_num*(17))

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
    rnd.seed(experiment_config.seed_num*(17))
    rnd.shuffle(all_mix_tuples)
    return split_list(all_mix_tuples, experiment_config.num_clients, experiment_config.seed*(1357))




def get_data_to_mix_and_data_to_leave(classes_data_dict):
    data_to_mix_list = []
    clients_original_data_dict = {}

    for class_name, single_tensor_set in classes_data_dict.items():
        size_to_mix = int(len(single_tensor_set) * experiment_config.mix_percentage)
        size_to_leave = int(len(single_tensor_set) * (1 - experiment_config.mix_percentage))

        torch.Generator().manual_seed(experiment_config.seed_num*(777))
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


    rnd.seed(experiment_config.seed_num*(19))
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

        rnd.seed(experiment_config.seed_num*(658+17*(i+1)))
        rnd.shuffle(data_per_client_list)
        ans.append(transform_to_TensorDataset(data_per_client_list))
    return ans

def group_labels_cifar100(label_dict, number_of_optimal_clusters):
    # Sort the labels to ensure sequential order
    labels = sorted(label_dict.keys())
    num_in_group = len(labels)/number_of_optimal_clusters

    # Create the groups dynamically based on the desired group_size
    groups = {
        f"group_{i // num_in_group + 1}": labels[i:i + int(num_in_group)]
        for i in range(0, len(labels), int(num_in_group))
    }
    ans = {}
    for group_name,group_lists in groups.items():
        data_per_group = []
        for member in group_lists:
            data_per_group.append(label_dict[member])
        ans[group_name] = data_per_group
    return ans

def group_labels(label_dict, number_of_optimal_clusters):
    # Sort the labels to ensure sequential order
    labels = sorted(label_dict.keys())
    num_in_group = len(labels) / number_of_optimal_clusters

    groups = {
        f"group_{i // num_in_group + 1}": labels[i:i + int(num_in_group)]
        for i in range(0, len(labels), int(num_in_group))
    }

    ans = {}
    for group_name,group_lists in groups.items():
        data_per_group = []
        for member in group_lists:
            data_per_group.append(label_dict[member])
        ans[group_name] = data_per_group
    return ans




def get_image_split_list_classes(tensor_list):
    image_split_list_classes=[]
    i=0
    image_list = []

    for tensor_col in tensor_list:
        i = i + 1

        for image in tensor_col:
            image_list.append(image)
    rnd.seed(experiment_config.seed_num*(543 + 5235 * (i + 3)))
    rnd.shuffle(image_list)
    chunk_size = len(image_list) // experiment_config.identical_clients

    image_split_list = [image_list[i * chunk_size: (i + 1) * chunk_size] for i in range(experiment_config.identical_clients)]
    image_split_list_classes.append(image_split_list)



    return image_split_list_classes


def get_image_split_list_classes_dich(tensor_list, num_clients, alpha,i):
    """
    Splits images of each class among clients using Dirichlet distribution (non-IID).

    Args:
        tensor_list (list): Each entry is a list/tensor of images for a specific class (total C classes).
        num_clients (int): Number of clients to split across.
        alpha (float): Dirichlet concentration parameter.

    Returns:
        image_split_list_classes (list): List of length num_clients, where each element is the image list for that client.
    """

    image_split_list_classes = [[] for _ in range(num_clients)]  # One list per client
    num_classes = len(tensor_list)


    for class_id, class_data in enumerate(tensor_list):
        np.random.seed(experiment_config.seed_num*(17+(i+1)*17+class_id) ) # For reproducibility

        data = list(class_data)  # Convert to list for shuffling
        rnd.seed(experiment_config.seed_num*(13+(i+1)*123+class_id))
        rnd.shuffle(data)

        # Dirichlet distribution over clients (for this class)
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Calculate how many images each client gets from this class
        class_size = len(data)
        split_sizes = (proportions * class_size).astype(int)

        # Fix rounding to ensure total adds up to class_size
        while split_sizes.sum() < class_size:
            split_sizes[np.argmin(split_sizes)] += 1
        while split_sizes.sum() > class_size:
            split_sizes[np.argmax(split_sizes)] -= 1

        # Assign to clients
        start_idx = 0
        for client_id, size in enumerate(split_sizes):
            end_idx = start_idx + size
            image_split_list_classes[client_id].extend(data[start_idx:end_idx])
            start_idx = end_idx

    return image_split_list_classes

def get_data_per_client_dict_and_mix_list(mix_list,target_original_data_dict,data_per_client_dict):
    i = 0
    for group_name, tensor_list in target_original_data_dict.items():
        all_images_per_group = []

        data_per_client_list = []
        i = i + 1
        #if experiment_config.alpha_dicht == DataDistTypes.NaiveNonIID:
        #    image_split_list_classes = get_image_split_list_classes(tensor_list)

        #else:
        image_split_list_classes=[]
        num_c = int(experiment_config.num_clients / experiment_config.number_of_optimal_clusters)
        image_split_list_classes.append( get_image_split_list_classes_dich(tensor_list,num_c,experiment_config.alpha_dich,i))

        torch.manual_seed(experiment_config.seed_num*(42))

        for identical_clients_index in range(len(image_split_list_classes[0])):

            for amount_of_classes in range(len(image_split_list_classes)):

                tt = []
                images = image_split_list_classes[amount_of_classes][identical_clients_index]
                size_to_mix = int(len(images) * experiment_config.mix_percentage)
                size_to_leave = int(len(images) * (1 - experiment_config.mix_percentage))
                try:
                    splits = random_split(images, [size_to_mix, size_to_leave])
                except:
                    try:
                        splits = random_split(images, [size_to_mix, size_to_leave+1])
                    except:
                        splits = random_split(images, [size_to_mix, size_to_leave-1])


                tt.extend(splits[1])
                mix_list.extend(splits[0])
            data_per_client_list.append(tt)
        data_per_client_dict[group_name] = data_per_client_list
    rnd.shuffle(mix_list)


    mix_list = split_list(mix_list, experiment_config.num_clients, experiment_config.seed_num*(1357))
    return mix_list


def get_clients_non_iid_data(target_original_data_dict):
    mix_list = []
    data_per_client_dict = {}

    mix_list = get_data_per_client_dict_and_mix_list(mix_list,target_original_data_dict,data_per_client_dict)

    i = 0
    for group_name,image_lists in data_per_client_dict.items():
        for image_list in image_lists:
            try:
                image_list.extend(mix_list[i])
                rnd.seed(experiment_config.seed_num*(523 + 412 * (i + 7)))
                rnd.shuffle(image_list)
            except:
                break
            i =i+1

    #clients_tensor_list = get_mix_tensor_list(target_original_data_dict)

    # if len(mix_tensor_list)!=len(clients_tensor_list):raise Exception("lists need to be same length")
    # match_mix_clients = get_match_mix_clients(mix_tensor_list,clients_non_iid_data)
    # ans = {}
    # for i in range(len(match_mix_clients)):ans[i]=match_mix_clients[i]

    # return ans

    return data_per_client_dict


def get_split_train_client_datav2(classes_data_dict):
    data_to_mix_list,target_original_data_dict = get_data_to_mix_and_data_to_leave(classes_data_dict)
    mix_tensor_list = get_mix_torch(data_to_mix_list)
    clients_tensor_list = get_mix_tensor_list(target_original_data_dict)
    if len(mix_tensor_list)!=len(clients_tensor_list):raise Exception("lists need to be same length")
    match_mix_clients = get_match_mix_clients(mix_tensor_list,clients_tensor_list)
    ans = {}
    for i in range(len(match_mix_clients)):ans[i]=match_mix_clients[i]

    return ans

def get_split_train_client_data(classes_data_dict):

    number_of_optimal_clusters =experiment_config.number_of_optimal_clusters
    target_original_data_dict = group_labels(classes_data_dict,number_of_optimal_clusters)
    clients_non_iid_data = get_clients_non_iid_data(target_original_data_dict)
    ans = {}
    for group_name, images_list in clients_non_iid_data.items():
        ans[group_name] = []

        for image_list in images_list:
            ans[group_name].append(transform_to_TensorDataset(image_list))
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

            rnd.seed(experiment_config.seed_num*(counter*17))
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



def cut_data_for_partial_use_of_data(class_target,data_by_classification_dict):
    data_of_class = data_by_classification_dict[class_target]
    train_set_size = len(data_of_class)
    size_use = int(experiment_config.percent_train_data_use * train_set_size)
    return cut_data(data_of_class, size_use)

def get_server_data(server_data_dict):
    server_data = create_server_data(server_data_dict)
    rnd.seed(experiment_config.seed_num*(42))
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


    total_client_data_size = int((1-server_split_ratio) * len(train_set))
    server_data_size = len(train_set) - total_client_data_size
    client_data_size = total_client_data_size // experiment_config.num_clients  # Each client gets an equal share
    split_sizes = [client_data_size] * experiment_config.num_clients  # List of client dataset sizes
    split_sizes.append(server_data_size)  # Add the remaining data for the server
    seed = 42
    torch.manual_seed(experiment_config.seed_num*(seed))
    splits = random_split(train_set, split_sizes)
    client_data_sets = splits[:-1]  # All client datasets

    #cut_data(total_client_data_size, experiment_config.percent_train_data_use*)

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


def get_clients_mix_data(clients_data_dict):
    ans = {}
    for group_num,clients_data_list in clients_data_dict.items():
        ans[group_num] =[]
        for tens_data in clients_data_list:
            for image in tens_data:
                ans[group_num].append(image)
    for group_num,images in ans.items():
        rnd.seed(experiment_config.seed_num*(42))
        rnd.shuffle(images)
        ans[group_num] = transform_to_TensorDataset(images)
    return ans

def download_and_extract_tiny_imagenet(data_dir='./data'):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(data_dir, 'tiny-imagenet-200.zip')
    extract_path = os.path.join(data_dir, 'tiny-imagenet-200')

    if os.path.exists(extract_path):
        print('Tiny ImageNet already downloaded and extracted.')
        return

    os.makedirs(data_dir, exist_ok=True)
    print('Downloading Tiny ImageNet...')
    urlretrieve(url, zip_path)

    print('Extracting...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print('Download and extraction complete.')

    # Optional: Clean up zip file to save space
    os.remove(zip_path)


def reorganize_tiny_imagenet_val(val_dir, root_dir):
    val_images_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'images')
    val_annotations_file = os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'val_annotations.txt')
    target_base = os.path.join(root_dir, 'tiny-imagenet-200', 'val')

    if not os.path.exists(val_annotations_file):
        return  # Nothing to reorganize

    with open(val_annotations_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            img_name = parts[0]
            label = parts[1]
            label_dir = os.path.join(target_base, label, 'images')
            os.makedirs(label_dir, exist_ok=True)
            shutil.move(os.path.join(val_images_dir, img_name), os.path.join(label_dir, img_name))

    shutil.rmtree(val_images_dir)
    os.remove(val_annotations_file)

import random
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

def default_augmentation_pipeline(img_size: Optional[Tuple[int, int]] = None):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    ])

def _gather_tensors_from_donors(datasets: List[TensorDataset], exclude_idx: int):
    donor_xs, donor_ys = [], []
    for j, ds in enumerate(datasets):
        if j == exclude_idx:
            continue
        xj, yj = ds.tensors
        donor_xs.append(xj)
        donor_ys.append(yj)
    if donor_xs:
        return torch.cat(donor_xs, dim=0), torch.cat(donor_ys, dim=0)
    return torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

@torch.no_grad()
def _augment_from_base(base_x, base_y, n_needed, aug, gen: torch.Generator):
    if base_x.numel() == 0 or n_needed <= 0:
        return torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=base_y.dtype)

    C, H, W = base_x.shape[1:]
    out_x = torch.empty((n_needed, C, H, W), dtype=torch.float32)
    out_y = torch.empty((n_needed,), dtype=base_y.dtype)

    idxs = torch.randint(0, base_x.shape[0], (n_needed,), generator=gen)
    for i, idx in enumerate(idxs):
        xi = aug(base_x[int(idx)])
        out_x[i] = xi
        out_y[i] = base_y[int(idx)]
    return out_x, out_y

def ensure_min_k_per_client(
    clients_data_dict: Dict[str, List[TensorDataset]],
    k: int,
    augmentation: Optional[transforms.Compose] = None,
    seed: Optional[int] = None
) -> Dict[str, List[TensorDataset]]:

    if augmentation is None:
        augmentation = default_augmentation_pipeline()

    # Local RNGs
    py_rng = random.Random(seed) if seed is not None else random.Random()
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    balanced = {}

    for group_key, datasets in clients_data_dict.items():
        new_list = []
        for i, ds in enumerate(datasets):
            xi, yi = ds.tensors
            xi, yi = xi.clone(), yi.clone()

            n = int(xi.shape[0])
            target_k = int(k)

            if n >= target_k:
                if n > target_k:
                    keep = torch.randperm(n, generator=gen)[:target_k]
                    xi, yi = xi[keep], yi[keep]
                new_list.append(TensorDataset(xi, yi))
                continue

            needed = int(target_k - n)
            donor_x, donor_y = _gather_tensors_from_donors(datasets, i)
            donor_n = int(donor_x.shape[0])

            # Borrow from donors
            if needed > 0 and donor_n > 0:
                take = int(min(needed, donor_n))
                donor_idxs = list(range(donor_n))
                py_rng.shuffle(donor_idxs)
                pick = donor_idxs[:take]                 # <- take is a plain int now
                xi = torch.cat([xi, donor_x[pick]], dim=0)
                yi = torch.cat([yi, donor_y[pick]], dim=0)
                needed -= take

            # Augmentation if still needed
            if needed > 0:
                base_x, base_y = ds.tensors
                X_aug, Y_aug = _augment_from_base(base_x, base_y, int(needed), augmentation, gen)
                xi = torch.cat([xi, X_aug], dim=0)
                yi = torch.cat([yi, Y_aug], dim=0)
                needed = 0

            # Final truncate to exactly k
            cur_n = int(xi.shape[0])
            if cur_n > target_k:
                keep = torch.randperm(cur_n, generator=gen)[:target_k]
                xi, yi = xi[keep], yi[keep]

            new_list.append(TensorDataset(xi, yi))

        balanced[group_key] = new_list

    return balanced


def download_and_extract_caltech256(destination_path='./data'):
    dataset_url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar'
    tar_path = os.path.join(destination_path, '256_ObjectCategories.tar')
    extract_path = os.path.join(destination_path, '256_ObjectCategories')

    os.makedirs(destination_path, exist_ok=True)

    # Download the dataset
    print(f'Downloading Caltech-256 dataset from {dataset_url}...')
    urllib.request.urlretrieve(dataset_url, tar_path)
    print('Download complete.')

    # Extract the dataset
    print('Extracting the dataset...')
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=destination_path)
    print('Extraction complete.')

    # Optionally, remove the tar file to save space
    os.remove(tar_path)
    print('Cleanup complete.')

    return extract_path

def get_data_set(is_train ):
    dataset = experiment_config.data_set_selected
    print(f"Loading dataset: {dataset}")

    if dataset == DataSet.EMNIST_balanced:


        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale to RGB
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Use RGB normalization
        ])
    elif dataset == DataSet.IMAGENET:
        if is_train:
            transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
            ])
    else:
        print("Using RGB transform for dataset")
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if experiment_config.data_set_selected == DataSet.CIFAR100:
        train_set = torchvision.datasets.CIFAR100(
            root='./data/cifar100',  # dedicated subdir
            train=is_train, download=True, transform=transform
        )
        experiment_config.target_k = 40000/25
    if experiment_config.data_set_selected == DataSet.IMAGENET:
        imagenet_dir = '/mnt/myssd/Ben/imagenet/train' if is_train else '/mnt/myssd/Ben/imagenet/val'
        train_set = torchvision.datasets.ImageFolder(root=imagenet_dir, transform=transform)

    if experiment_config.data_set_selected == DataSet.CIFAR10:
        train_set = torchvision.datasets.CIFAR10(root='./data', train=is_train, download=True, transform=transform)

    if dataset == DataSet.SVHN:
        split = 'train' if is_train else 'test'
        train_set = torchvision.datasets.SVHN(root='./data', split=split, download=True, transform=transform)

    if experiment_config.data_set_selected == DataSet.TinyImageNet:
        download_and_extract_tiny_imagenet('./data')
        data_root = './data/tiny-imagenet-200'
        subfolder = 'train' if is_train else 'val'
        dataset_path = os.path.join(data_root, subfolder)

        # Reorganize val/ folder if needed
        if subfolder == 'val':
            reorganize_tiny_imagenet_val(dataset_path, './data')

        train_set = ImageFolder(
            root=dataset_path,
            transform=transform
        )



    if experiment_config.data_set_selected == DataSet.EMNIST_balanced:
        train_set = EMNIST(root='./data', split='balanced', train=is_train, download=True,
           transform=transform)

    # Use memory-efficient classification for ImageNet to prevent memory explosion
    if experiment_config.data_set_selected == DataSet.IMAGENET:
        print("Using memory-efficient data classification for ImageNet...")
        data_by_classification_dict = get_data_by_classification_memory_efficient(train_set)
        selected_classes_list = sorted(data_by_classification_dict.keys())[:experiment_config.num_classes]
        
        # Use memory-efficient splitting
        clients_data_dict, server_data = split_clients_server_data_memory_efficient(
            data_by_classification_dict, selected_classes_list, train_set
        )
    else:
        # Use original method for other datasets
        data_by_classification_dict = get_data_by_classification(train_set)
        selected_classes_list = sorted(data_by_classification_dict.keys())[:experiment_config.num_classes]
        clients_data_dict, server_data = split_clients_server_data_Non_IID(data_by_classification_dict, selected_classes_list)



    if experiment_config.num_clients > 25:
        balanced = ensure_min_k_per_client(
            clients_data_dict,
            k= (40000/25),
            seed=experiment_config.seed_num  # reproducible donor sampling & augmentation

        )
        clients_data_dict = balanced

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


def create_data():


    selected_classes_list, clients_train_data_dict, server_train_data = get_data_set( is_train = True)
    selected_test_classes_list, clients_test_data_dict, server_test_data = get_data_set( is_train = False)



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

    torch.Generator().manual_seed(experiment_config.seed_num*(999))
    splits = random_split(train_set_images, split_sizes)
    clients_data = transform_to_TensorDataset(splits[0])  # All client datasets
    server_data = transform_to_TensorDataset(splits[1])

    return clients_data, server_data



#### ----------------- CREATE CLIENTS ----------------- ####


def get_random_dataset(dataset, percent = experiment_config.percent_train_data_use):

    total_size = len(dataset)
    subset_size = int(total_size * (percent / 100))

    # Get random indices
    indices = rnd.sample(range(total_size), subset_size)

    # Extract the samples using the indices
    data = [dataset[i] for i in indices]

    # Create a new dataset of the same type
    if isinstance(dataset, torch.utils.data.TensorDataset):
        # Handle TensorDataset separately
        tensors = [torch.stack([sample[i] for sample in data]) for i in range(len(dataset.tensors))]
        return torch.utils.data.TensorDataset(*tensors)

    # Otherwise, assume it's a standard Dataset and return a custom wrapper
    class RandomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return RandomDataset(data)

def cut_dict(data_dict: {}):
    ans = {}
    for k, v in data_dict.items():
        ans[k] = get_random_dataset(v)
    return ans

def cut_data_v2(clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data):
    clients_train_data_dict = cut_dict(clients_train_data_dict)
    server_train_data = get_random_dataset(server_train_data)
    clients_test_data_dict = cut_dict(clients_test_data_dict)
    server_test_data = get_random_dataset(server_test_data)
    return clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data

def divide_list(lst, x):
    """Divide a list into x groups as equally as possible."""
    rnd.seed(experiment_config.seed_num)
    rnd.shuffle(lst)

    avg = len(lst) // x
    remainder = len(lst) % x
    groups = []
    start = 0

    for i in range(x):
        end = start + avg + (1 if i < remainder else 0)
        groups.append(lst[start:end])
        start = end

    return groups

def fix_global_data(server_train_data):
    torch.manual_seed(experiment_config.seed_num)
    data = server_train_data
    x = experiment_config.iterations
    # Compute split sizes
    images = []
    for image in data:
        images.append(image)
    divided_list = divide_list(images,x)
    ans = []
    for lst in divided_list:
        ans.append(transform_to_TensorDataset(lst))
    return ans
    #split_sizes = [len(data) // x] * x
    #for i in range(len(data) % x):  # Handle remainder
    #    split_sizes[i] += 1

    # Split the dataset
    #return  random_split(data, split_sizes)





def create_clients(client_data_dict,server_data,test_set,server_test_data):
    clients_test_by_id_dict = {}
    ans = []
    ids_list = []
    id_ = 0
    known_clusters = {}
    cluster_num = -1

    for group_name,data_list in client_data_dict.items():
        cluster_num = cluster_num+1
        known_clusters[cluster_num] = []

        data_index = 0

        for data_ in data_list:
            ids_list.append(id_)
            known_clusters[cluster_num].append(id_)
            if experiment_config.algorithm_selection ==AlgorithmSelected.PseudoLabelsClusters:
                c = Client(id_=id_, client_data=data_, global_data=server_data, global_test_data=server_test_data,
                              local_test_data=test_set[group_name][data_index])

            if  experiment_config.algorithm_selection == AlgorithmSelected.PseudoLabelsNoServerModel or experiment_config.algorithm_selection == AlgorithmSelected.COMET:
                c = Client(id_=id_, client_data=data_, global_data=server_data, global_test_data=server_test_data,
                           local_test_data=test_set[group_name][data_index])
            if experiment_config.algorithm_selection ==AlgorithmSelected.NoFederatedLearning:

                c = Client_NoFederatedLearning(id_=id_, client_data=data_, global_data=server_data, global_test_data=server_test_data,
                           local_test_data=test_set[group_name][data_index],evaluate_every=experiment_config.epochs_num_input_fine_tune_clients)
            if experiment_config.algorithm_selection == AlgorithmSelected.PseudoLabelsClusters_with_division:
                c = Client_PseudoLabelsClusters_with_division(id_=id_, client_data=data_, global_data=server_data,
                                               global_test_data=server_test_data,
                                               local_test_data=test_set[group_name][data_index])
            if experiment_config.algorithm_selection== AlgorithmSelected.FedAvg:
                c = Client_FedAvg(id_=id_, client_data=data_, global_data=server_data,
                                                              global_test_data=server_test_data,
                                                              local_test_data=test_set[group_name][data_index])
            if experiment_config.algorithm_selection == AlgorithmSelected.pFedCK:
                c = Client_pFedCK(id_=id_, client_data=data_, global_data=server_data, global_test_data=server_test_data,
                           local_test_data=test_set[group_name][data_index])

            ans.append(c)
            clients_test_by_id_dict[id_] = test_set[group_name][data_index]
            data_index = data_index+1
            id_ = id_+1
    experiment_config.known_clusters = known_clusters




    return ans,ids_list,clients_test_by_id_dict



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

def create_tensor_dataset_from_indices(dataset, indices, batch_size=None):
    """
    Create TensorDataset from indices in batches to avoid memory overflow.
    """
    if batch_size is None:
        batch_size = getattr(experiment_config, 'tensor_creation_batch_size', 1000)
    
    images = []
    targets = []
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = [dataset[idx] for idx in batch_indices]
        
        batch_images = [item[0] for item in batch_data]
        batch_targets = [item[1] for item in batch_data]
        
        images.extend(batch_images)
        targets.extend(batch_targets)
    
    images_tensor = torch.stack(images)
    targets_tensor = torch.tensor(targets)
    
    return TensorDataset(images_tensor, targets_tensor)

def split_clients_server_data_memory_efficient(data_by_classification_dict, selected_classes, dataset):
    """
    Memory-efficient version that works with indices instead of full data.
    """
    clients_data_dict = {}
    server_data_dict = {}
    
    for class_name in selected_classes:
        indices = data_by_classification_dict[class_name]
        
        # Split indices for clients and server
        total_size = len(indices)
        server_size = int(total_size * experiment_config.server_split_ratio)
        client_size = total_size - server_size
        
        # Shuffle indices
        torch.Generator().manual_seed(experiment_config.seed_num * (999 + hash(class_name)))
        shuffled_indices = torch.randperm(total_size).tolist()
        shuffled_indices = [indices[i] for i in shuffled_indices]
        
        # Split indices
        client_indices = shuffled_indices[:client_size]
        server_indices = shuffled_indices[client_size:]
        
        # Create datasets from indices
        clients_data_dict[class_name] = create_tensor_dataset_from_indices(dataset, client_indices)
        server_data_dict[class_name] = create_tensor_dataset_from_indices(dataset, server_indices)
    
    # Combine server data
    all_server_indices = []
    for indices in server_data_dict.values():
        all_server_indices.extend(indices)
    
    server_data = create_tensor_dataset_from_indices(dataset, all_server_indices)
    
    return clients_data_dict, server_data