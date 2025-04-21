from enum import Enum
from random import random
from xml.dom import NoDataAllowedErr

import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset

cifar100_label_to_superclass = {
    0: 0,  1: 0,  2: 0,  3: 0,  4: 0,     # Aquatic mammals
    5: 1,  6: 1,  7: 1,  8: 1,  9: 1,     # Fish
    10: 2, 11: 2, 12: 2, 13: 2, 14: 2,    # Flowers
    15: 3, 16: 3, 17: 3, 18: 3, 19: 3,    # Food containers
    20: 4, 21: 4, 22: 4, 23: 4, 24: 4,    # Fruit and vegetables
    25: 5, 26: 5, 27: 5, 28: 5, 29: 5,    # Household electrical devices
    30: 6, 31: 6, 32: 6, 33: 6, 34: 6,    # Household furniture
    35: 7, 36: 7, 37: 7, 38: 7, 39: 7,    # Insects
    40: 8, 41: 8, 42: 8, 43: 8, 44: 8,    # Large carnivores
    45: 9, 46: 9, 47: 9, 48: 9, 49: 9,    # Large man-made outdoor things
    50: 10, 51: 10, 52: 10, 53: 10, 54: 10,  # Large natural outdoor scenes
    55: 11, 56: 11, 57: 11, 58: 11, 59: 11,  # Large omnivores and herbivores
    60: 12, 61: 12, 62: 12, 63: 12, 64: 12,  # Medium-sized mammals
    65: 13, 66: 13, 67: 13, 68: 13, 69: 13,  # Non-insect invertebrates
    70: 14, 71: 14, 72: 14, 73: 14, 74: 14,  # People
    75: 15, 76: 15, 77: 15, 78: 15, 79: 15,  # Reptiles
    80: 16, 81: 16, 82: 16, 83: 16, 84: 16,  # Small mammals
    85: 17, 86: 17, 87: 17, 88: 17, 89: 17,  # Trees
    90: 18, 91: 18, 92: 18, 93: 18, 94: 18,  # Vehicles 1
    95: 19, 96: 19, 97: 19, 98: 19, 99: 19   # Vehicles 2
}




def transform_to_TensorDataset_v2(data_):
    images = []
    targets = []

    for item in data_:
        if isinstance(item, tuple) and len(item) == 2:
            img, label = item
        else:
            raise ValueError(f"Expected tuple (image, label), but got: {type(item)} with value {item}")

        # Convert 0-dim tensors to values if needed
        if isinstance(label, torch.Tensor) and label.dim() == 0:
            label = label.item()

        images.append(img)
        targets.append(label)

    # Stack into tensors
    images_tensor = torch.stack(images)
    targets_tensor = torch.tensor(targets)

    return TensorDataset(images_tensor, targets_tensor)

def transform_to_TensorDataset(data_):
    images = [item[0] for item in data_]  # Extract the image tensors (index 0 of each tuple)
    targets = [item[1] for item in data_]

    # Step 2: Convert the lists of images and targets into tensors (if not already)
    images_tensor = torch.stack(images)  # Stack the image tensors into a single tensor
    targets_tensor = torch.tensor(targets)  # Convert the targets to a tensor


    # Step 3: Create a TensorDataset from the images and targets
    return TensorDataset(images_tensor, targets_tensor)


class ServerInputTech(Enum):
    mean = 1
    max = 2
class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

class DataSet(Enum):
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"


class DataType(Enum):
    IID = 1
    NonIID = 2

class NetsType(Enum):
    C_alex_S_vgg = 1
    C_alex_S_alex = 2
    C_alex = 3
    S_alex = 4
    S_vgg = 5


class ClusterTechnique(Enum):
    kmeans = 1
    manual_L2 = 2
    manual_cross_entropy = 3

    manual_single_iter = 4
    greedy_elimination_cross_entropy = 5
    greedy_elimination_L2 = 6

class ServerFeedbackTechnique(Enum):
    similar_to_client = 1
    similar_to_cluster = 2

class NetClusterTechnique(Enum):
    multi_head = 1
    multi_model = 2
    no_model=3
class AlgorithmSelected(Enum):
    PseudoLabelsClusters = 1
    PseudoLabelsNoServerModel = 2
    NoFederatedLearning = 3
    PseudoLabelsClusters_with_division = 4
    Centralized = 5
    FedAvg = 6
class DataDistTypes(Enum):
    NaiveNonIID = 1
    Dirichlet1 = 2

class ExperimentConfig:
    def __init__(self):
        self.which_net_arch = None
        self.seed_num = 1
        self.iterations = 20

        # CIFAR10/CIFAR 100
        self.data_set_selected = None # selected in main
        self.num_classes = None

        # scale and complexity
        self.num_clients = None  # selected in main
        self.number_of_optimal_clusters = None # selected in main
        self.identical_clients = None # num_clients / number_of_optimal_clusters
        self.mix_percentage = None
        self.server_split_ratio = 0.2


        # net types:
        self.client_net_type = None
        self.server_net_type = None
        self.learning_rate_train_c = None
        self.learning_rate_fine_tune_c = None
        self.learning_rate_train_s = None

        #Algo typy
        self.algorithm_selection = None



        self.net_cluster_technique = None
        self.server_input_tech = None
        self.server_feedback_technique = None
        self.cluster_technique = None


        #epochs
        self.epochs_num_train_server = 5
        self.epochs_num_input_fine_tune_clients = 5
        self.epochs_num_train_client = 5
        self.epochs_num_input_fine_tune_clients_no_fl = self.epochs_num_input_fine_tune_clients*self.iterations
        self.epochs_num_input_fine_tune_centralized_server = self.epochs_num_input_fine_tune_clients*self.iterations
        self.alpha_dich = 100



        # general vars
        self.local_batch = 64
        self.batch_size = 32
        self.percent_train_data_use = 1
        self.percent_test_relative_to_train = 1
        self.num_rounds_multi_head = 1
        self.known_clusters = None


        #




        self.num_clusters = None
        self.percent_train_data_use = 1
        self.percent_test_relative_to_train = 1

        self.cluster_addition = None


    def update_num_classes(self,data_set):
        self.data_set_selected = data_set
        if data_set == DataSet.CIFAR100:
            self.num_classes = 100
        if data_set == DataSet.CIFAR10:
            self.num_classes = 10

    def to_dict(self):
        """Returns a dictionary of attribute names and their values."""
        return {attr: getattr(self, attr) for attr in dir(self) if
                not callable(getattr(self, attr)) and not attr.startswith("__")}


    def update_net_type(self,net_type):
        if net_type == NetsType.C_alex_S_alex or net_type == NetsType.C_alex or net_type==NetsType.S_alex or net_type==NetsType.S_vgg:
            self.client_net_type = NetType.ALEXNET
            self.server_net_type = NetType.ALEXNET
            self.learning_rate_train_c = 0.0001
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.001







        if net_type == NetsType.C_alex_S_vgg or net_type == NetsType.S_vgg:
            self.client_net_type = NetType.ALEXNET
            self.server_net_type = NetType.VGG
            self.learning_rate_train_c = 0.0001
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.0001















experiment_config = ExperimentConfig()






