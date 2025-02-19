from enum import Enum
from random import random

from matplotlib import pyplot as plt

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

class ServerInputTech(Enum):
    mean = 1
    max = 2
class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

class DataSet(Enum):
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"



class NetsType(Enum):
    C_alex_S_vgg = 1
    C_alex_S_alex = 2
    C_alex_S_None = 3






class ClusterTechnique(Enum):
    kmeans = 1
    manual = 2

class ServerFeedbackTechnique(Enum):
    similar_to_client = 1
    similar_to_cluster = 2

class NetClusterTechnique(Enum):
    multi_head = 1
    multi_model = 2

class ExperimentConfig:
    def __init__(self):
        self.data_set_selected = None  # update in main
        self.num_classes = None # depends on data_set_selected 100

        self.num_clients = None  # update in main
        self.identical_clients = None# int( experiment_config.num_clients / experiment_config.number_of_optimal_clusters)
        self.mix_percentage = None # set in main


        #net type variables depends on the the type variable
        self.client_net_type = None
        self.server_net_type = None
        self.learning_rate_train_c = None
        self.learning_rate_fine_tune_c = None
        self.learning_rate_train_s = None
        self.epochs_num_train_server = None
        self.epochs_num_input_fine_tune_clients = None
        self.epochs_num_train_client = None
        self.num_clusters = None
        self.cluster_technique = None


        self.iterations = 11

        #
        self.net_cluster_technique = NetClusterTechnique.multi_model
        self.server_input_tech = ServerInputTech.max
        self.server_feedback_technique = ServerFeedbackTechnique.similar_to_cluster
        self.num_rounds_multi_head =  1
        self.batch_size = 128
        self.seed_num = 1
        self.server_split_ratio = 0.2
        self.percent_train_data_use = 1
        self.percent_test_relative_to_train = 1
        self.number_of_optimal_clusters = 10

        self.known_clusters = None

    def to_dict(self):
        """Returns a dictionary of attribute names and their values."""
        return {attr: getattr(self, attr) for attr in dir(self) if
                not callable(getattr(self, attr)) and not attr.startswith("__")}


    def update_net_type(self,net_type):
        if net_type == NetsType.C_alex_S_alex:
            self.client_net_type = NetType.ALEXNET
            self.server_net_type = NetType.ALEXNET
            self.learning_rate_train_c = 0.001
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.001
            self.epochs_num_train_server = 10
            self.epochs_num_input_fine_tune_clients = 10
            self.epochs_num_train_client = 10


        if net_type == NetsType.C_alex_S_vgg:
            self.client_net_type = NetType.ALEXNET
            self.server_net_type = NetType.VGG
            self.learning_rate_train_c = 0.001
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.0001
            self.epochs_num_train_server = 10
            self.epochs_num_input_fine_tune_clients = 10
            self.epochs_num_train_client = 10







        #######-------------------------------------
        #self.update_data_type(data_type)

        #######-------------------------------------
        #self.update_net_type(net_type)

        ##############------------------------------------------------














experiment_config = ExperimentConfig()






