from enum import Enum
from random import random

import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, Subset
from torchvision.models import MobileNetV2

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
    """
    If data_ is (base_dataset, indices) -> return Subset(base_dataset, indices)
    Else assume an iterable of (image_tensor, label) -> return TensorDataset
    """
    # New path: zero-copy view over the original dataset
    if isinstance(data_, tuple) and len(data_) == 2:
        base_dataset, indices = data_
        return Subset(base_dataset, list(indices))

    # Legacy path: materialized list of (tensor, label)
    images = [item[0] for item in data_]
    targets = [item[1] for item in data_]
    return TensorDataset(torch.stack(images), torch.tensor(targets))


class ServerInputTech(Enum):
    mean = 1
    max = 2
    median =3

class NetType(Enum):
    AlexSqueeze = "AlexSqueeze"
    AlexMobile = "AlexMobile"
    ResMobile = "ResNetMobile"
    ResNetSqueeze = "ResNetSqueeze"
    AlexMobileResnet = "AlexMobileResnet"


    SqueezeNet = "SqueezeNet"
    MobileNet = "MobileNet"
    rndNet = "rndNet"

    ALEXNET = "AlexNet"
    VGG = "VGG"
    ResNet = "ResNet"
    DenseNetServer= "DenseNetServer"
    rndStrong = "rndStrong"
    rndWeak = "rndWeak"

class DataSet(Enum):
    ImageNetR = "ImageNetR"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    TinyImageNet = "TinyImageNet"
    EMNIST_balanced = "EMNIST_balanced"
    SVHN = "SVHN"
    ImageNet100 = "ImageNet100"


class DataType(Enum):
    IID = 1
    NonIID = 2

class NetsType(Enum):
    C_AlexSqueeze_S_vgg = 35
    C_AlexSqueeze_S_alex = 34
    C_AlexMobile_S_vgg = 33
    C_AlexMobile_S_alex = 32
    C_AlexMobileResnet_S_alex = 2000
    C_AlexMobileResnet_S_VGG = 2001

    C_ResNetMobile_S_vgg = 31
    C_ResNetMobile_S_alex = 30
    C_ResNetSqueeze_S_vgg = 29
    C_ResNetSqueeze_S_alex = 28

    C_squeeze_S_alex = 27

    C_squeeze_S_vgg = 26
    C_ResNet_S_alex = 25
    C_ResNet_S_vgg = 24
    C_alex_S_Mobile = 23
    C_Mobile_S_alex = 20
    C_Mobile_S_VGG = 21
    C_rnd_S_VGG = 14
    C_rnd_S_alex = 15

    C_alex_S_vgg = 1
    C_alex_S_alex = 2
    C_alex_S_ResNet = 6
    C_rndStrong_S_alex = 10
    C_rndStrong_S_VGG = 11

    C_rndWeak_S_alex = 12
    C_rndWeak_S_VGG = 13
    C_rndWeak_S_Mobile = 1000
    C_rndWeak_S_ResNet = 1001
    C_rndWeak_S_Squeeze = 1003


    C_alex_S_DenseNet = 7

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

class WeightForPS (Enum):
    withWeights = 1
    withoutWeights = 2

class InputConsistency(Enum):
    withInputConsistency = 1
    withoutInputConsistency = 2
class AlgorithmSelected(Enum):
    pFedHN = 13

    pFedMe = 11
    FedBABU = 10
    MAPL = 1
    FedMD = 2
    NoFederatedLearning = 3
    PseudoLabelsClusters_with_division = 4
    Centralized = 5
    FedAvg = 6
    pFedCK = 7
    COMET = 8
    Ditto = 9
    FedSelect = 12
    FedCT = 999


class DataDistTypes(Enum):
    NaiveNonIID = 1
    Dirichlet1 = 2

class ExperimentConfig:
    def __init__(self):

        self.server_learning_rate_mapl = 0.0001#_mapl = [0.005, 0.001, 0.0005, 0.0001, 0.00001]  # 0.005,
        self.client_lr_local_lr_distill_mapl = (1e-3, 1e-3)#_mapl = [(1e-3, 1e-3)]  # ,(1e-3, 1e-5),(1e-3, 1e-4),(1e-4, 1e-3),(1e-2, 1e-4)]
        self.distill_temperature = 1
        self.lambda_consistency = 1


        self.weights_for_ps = None
        self.input_consistency = None
        self.lambda_ditto = 1
        self.which_net_arch = None
        self.seed_num = 1
        self.iterations = 10
        self.is_with_memory_load = None
        self.beta = 0

        # CIFAR10/CIFAR 100
        self.data_set_selected = None # selected in main
        self.num_classes = None

        # scale and complexity
        self.num_clients = None  # selected in main
        self.number_of_optimal_clusters = None # selected in main
        self.identical_clients = None # num_clients / number_of_optimal_clusters
        self.mix_percentage = None
        self.server_split_ratio = 0.2
        self.server_data_ratio = None

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
        self.batch_size = 32#32
        self.percent_train_data_use = 1
        self.percent_test_relative_to_train = 1
        self.num_rounds_multi_head = 1
        self.known_clusters = None
        self.server_net_type_name = None
        self.client_net_type_name = None

        #




        self.num_clusters = None
        self.percent_train_data_use = 1
        self.percent_test_relative_to_train = 1

        self.cluster_addition = None


    def update_num_classes(self,data_set):
        self.data_set_selected = data_set
        if data_set == DataSet.CIFAR100:
            self.num_classes = 100
        if data_set == DataSet.ImageNetR:
            self.num_classes = 200
        if data_set == DataSet.CIFAR10:
            self.num_classes = 10
        if data_set == DataSet.TinyImageNet:
            self.num_classes = 200
            self.batch_size = 64  # 32
        if data_set == DataSet.EMNIST_balanced:
            self.num_classes = 47
            self.batch_size = 64  # 32

        if data_set == DataSet.SVHN:
            self.num_classes = 10

    def to_dict(self):
        """Returns a dictionary of attribute names and their values."""
        return {attr: getattr(self, attr) for attr in dir(self) if
                not callable(getattr(self, attr)) and not attr.startswith("__")}


    def update_net_type(self,net_type):


        if net_type ==NetsType.C_alex_S_Mobile:
            if net_type == NetsType.C_alex_S_Mobile:
                self.client_net_type = NetType.ALEXNET
                self.server_net_type = NetType.MobileNet
            if self.algorithm_selection == AlgorithmSelected.COMET:
                self.learning_rate_train_c = 0.0008
            elif self.algorithm_selection == AlgorithmSelected.FedMD:
                self.learning_rate_train_c = 0.002
            else:
                self.learning_rate_train_c = 0.0001

            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.001

        if net_type == NetsType.C_alex_S_alex or net_type == NetsType.C_alex or net_type==NetsType.S_alex or net_type==NetsType.S_vgg or  net_type == NetsType.C_rndStrong_S_alex or net_type == NetsType.C_rndWeak_S_alex or net_type == NetsType.C_rnd_S_alex or net_type ==NetsType.C_Mobile_S_alex  or  net_type == NetsType.C_ResNet_S_alex \
                or net_type == NetsType.C_squeeze_S_alex or net_type == NetsType.C_ResNetSqueeze_S_alex or net_type == NetsType.C_ResNetMobile_S_alex or net_type == NetsType.C_AlexMobile_S_alex or net_type == NetsType.C_AlexSqueeze_S_alex or net_type == NetsType.C_AlexMobileResnet_S_alex:

            if net_type == NetsType.C_rndStrong_S_alex:
                self.client_net_type = NetType.rndStrong
            elif net_type == NetsType.C_AlexMobile_S_alex:
                self.client_net_type = NetType.AlexMobile
            elif net_type == NetsType.C_AlexMobileResnet_S_alex:
                self.client_net_type = NetType.AlexMobileResnet
            elif net_type == NetsType.C_AlexSqueeze_S_alex:
                self.client_net_type = NetType.AlexSqueeze
            elif net_type == NetsType.C_ResNetMobile_S_alex:
                self.client_net_type = NetType.ResMobile
            elif net_type == NetsType.C_squeeze_S_alex:
                self.client_net_type = NetType.SqueezeNet
            elif net_type == NetsType.C_ResNetSqueeze_S_alex:
                self.client_net_type = NetType.ResNetSqueeze
            elif net_type == NetsType.C_ResNet_S_alex:
                self.client_net_type = NetType.ResNet
            elif net_type == NetsType.C_rndWeak_S_alex:
                self.client_net_type = NetType.rndWeak
            elif net_type == NetsType.C_rnd_S_alex:
                self.client_net_type = NetType.rndNet
            elif  net_type ==NetsType.C_Mobile_S_alex:
                self.client_net_type = NetType.MobileNet
            else:
                self.client_net_type = NetType.ALEXNET
            self.server_net_type = NetType.ALEXNET
            if self.algorithm_selection == AlgorithmSelected.COMET:
                self.learning_rate_train_c = 0.0008
            elif self.algorithm_selection == AlgorithmSelected.FedMD:
                self.learning_rate_train_c = 0.002
            else:
                self.learning_rate_train_c = 0.0001

            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.001



        if net_type == NetsType.C_rndWeak_S_Mobile or  net_type ==NetsType.C_rndWeak_S_ResNet  or net_type ==NetsType.C_rndWeak_S_Squeeze:
            if net_type ==NetsType.C_rndWeak_S_Mobile:
                self.server_net_type = NetType.MobileNet
            if net_type ==NetsType.C_rndWeak_S_ResNet:
                self.server_net_type = NetType.ResNet
            if net_type ==NetsType.C_rndWeak_S_Squeeze:
                self.server_net_type = NetType.SqueezeNet



            self.client_net_type = NetType.rndWeak
            self.learning_rate_train_c = 0.0001
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.0001
            if self.algorithm_selection == AlgorithmSelected.MAPL:
                #       self.server_learning_rate = 0.0001#_mapl = [0.005, 0.001, 0.0005, 0.0001, 0.00001]  # 0.005,
                # self.client_lr_local_lr_distill = (1e-3, 1e-3)#_mapl = [(1e-3, 1e-3)]  # ,(1e-3, 1e-5),(1e-3, 1e-4),(1e-4, 1e-3),(1e-2, 1e-4)]

                self.learning_rate_train_c = self.client_lr_local_lr_distill_mapl[1]
                self.learning_rate_fine_tune_c = self.client_lr_local_lr_distill_mapl[0]
                self.learning_rate_train_s = self.server_learning_rate_mapl


        if net_type == NetsType.C_alex_S_vgg or net_type == NetsType.S_vgg or net_type == NetsType.C_rndStrong_S_VGG or net_type == NetsType.C_rndWeak_S_VGG or net_type == NetsType.C_rnd_S_VGG or   net_type == NetsType.C_Mobile_S_VGG or  net_type == NetsType.C_ResNet_S_vgg or net_type == NetsType.C_squeeze_S_vgg or net_type == NetsType.C_ResNetSqueeze_S_vgg or net_type == NetsType.C_ResNetMobile_S_vgg or net_type == NetsType.C_AlexMobile_S_vgg or net_type == NetsType.C_AlexSqueeze_S_vgg  or net_type == NetsType.C_AlexMobileResnet_S_VGG:

            if net_type == NetsType.C_ResNet_S_vgg:
                self.client_net_type= NetType.ResNet
            elif net_type == NetsType.C_AlexMobileResnet_S_VGG:
                self.client_net_type = NetType.AlexMobileResnet
            elif net_type == NetsType.C_AlexMobile_S_vgg:
                self.client_net_type = NetType.AlexMobile
            elif net_type == NetsType.C_AlexSqueeze_S_vgg:
                self.client_net_type = NetType.AlexSqueeze
            elif net_type == NetsType.C_squeeze_S_vgg:
                self.client_net_type = NetType.SqueezeNet
            elif net_type == NetsType.C_ResNetMobile_S_vgg:
                self.client_net_type = NetType.ResMobile
            elif net_type == NetsType.C_ResNetSqueeze_S_vgg:
                self.client_net_type = NetType.ResNetSqueeze

            elif net_type == NetsType.C_rndStrong_S_VGG:
                self.client_net_type = NetType.rndStrong
            elif net_type == NetsType.C_rndWeak_S_VGG:
                self.client_net_type = NetType.rndWeak
            elif net_type == NetsType.C_rnd_S_VGG:
                self.client_net_type = NetType.rndNet
            elif  net_type == NetsType.C_Mobile_S_VGG:
                self.client_net_type = NetType.MobileNet
            else:
                self.client_net_type = NetType.ALEXNET
            self.server_net_type = NetType.VGG
            self.learning_rate_train_c = 0.0001
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.0001


            if self.algorithm_selection == AlgorithmSelected.MAPL:

                #       self.server_learning_rate = 0.0001#_mapl = [0.005, 0.001, 0.0005, 0.0001, 0.00001]  # 0.005,
        #self.client_lr_local_lr_distill = (1e-3, 1e-3)#_mapl = [(1e-3, 1e-3)]  # ,(1e-3, 1e-5),(1e-3, 1e-4),(1e-4, 1e-3),(1e-2, 1e-4)]

                self.learning_rate_train_c = self.client_lr_local_lr_distill_mapl[1]
                self.learning_rate_fine_tune_c = self.client_lr_local_lr_distill_mapl[0]
                self.learning_rate_train_s = self.server_learning_rate_mapl


            print("self.learning_rate_train_c",self.learning_rate_train_c )
            print("self.learning_rate_fine_tune_c",self.learning_rate_fine_tune_c )
            print("self.learning_rate_train_s",self.learning_rate_train_s )

        if  self.server_net_type is not None:
            self.server_net_type_name = ""
        else:
            self.server_net_type_name = self.server_net_type.name
        if self.algorithm_selection == AlgorithmSelected.COMET:
            self.learning_rate_train_c = 0.0008
        elif self.algorithm_selection == AlgorithmSelected.FedMD:
            self.learning_rate_train_c = 0.002

        self.client_net_type_name = self.client_net_type.name
        self.server_net_type_name = self.server_net_type.name

















experiment_config = ExperimentConfig()





