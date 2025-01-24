from enum import Enum
from random import random

from matplotlib import pyplot as plt


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
    C_alex_S_None = 3

class ExpType(Enum):
    full= 1
    short=2
    #IID_diff_nets = 1
    #NonIID_diff_nets = 2
    #IID_same_nets = 3
    #NonIID_same_nets = 4
    #IID_no_net = 5
    #NonIID_no_net = 6
    #short = 7



class ClusterTechnique(Enum):
    kmeans = 1
    manual = 2

class ServerFeedbackTechnique(Enum):
    similar_to_client = 1
    similar_to_cluster = 2


class ExperimentConfig:
    def __init__(self):
        self.identical_clients = None
        self.mix_percentage = None
        self.server_net_type = None
        self.client_net_type = None
        self.batch_size = None
        self.learning_rate_train_s = None

        self.cluster_technique = None
        self.server_feedback_technique = None

        self.data_set_selected = DataSet.CIFAR10
        self.num_classes = 10

        self.seed_num = 1
        self.with_weight_memory = True
        self.with_server_net = True
        self.epochs_num_input = 20
        self.epochs_num_train = 10

        self.iterations = 12
        self.server_split_ratio = 0.2
        self.learning_rate_fine_tune_c = 0.001
        self.learning_rate_train_c = 0.001
        self.num_clusters = None

        # ----------------

        # ----------------

        self.num_clients = None  # num_classes*identical_clients
        self.percent_train_data_use = 1
        self.percent_test_relative_to_train = 1

    def update_data_type(self,data_type):
        if data_type == DataType.IID:
            self.mix_percentage = 1
            self.identical_clients = 1
            self.batch_size = 128
            self.num_clients = 10

        if data_type == DataType.NonIID:
            self.mix_percentage = 0.2
            self.identical_clients = 2
            self.batch_size = 128
            self.num_clients = 10
            if self.num_clients%self.identical_clients!=0:
                raise Exception("identical clients must be dividable by identical clients i.e., self.num_clients%self.identical_clients=0? ")

    def update_net_type(self,net_type):
        if net_type == NetsType.C_alex_S_alex:
            self.client_net_type = NetType.ALEXNET
            self.server_net_type = NetType.ALEXNET
            self.learning_rate_train_c = 0.001
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.001
            self.with_server_net = True

        if net_type == NetsType.C_alex_S_vgg:
            self.client_net_type = NetType.ALEXNET
            self.server_net_type = NetType.VGG
            self.learning_rate_train_c = 0.001
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.0001
            self.with_server_net = True

        if net_type == NetsType.C_alex_S_None:
            self.client_net_type = NetType.VGG
            self.server_net_type = ""
            self.learning_rate_train_c = None
            self.learning_rate_fine_tune_c = 0.001
            self.learning_rate_train_s = 0.001
            self.with_server_net = False


    def update_type_of_experiment(self,exp_type):




        if exp_type == ExpType.short:
            self.num_classes = 10
            self.with_server_net = True
            #net_type = NetsType.C_alex_S_alex
            #data_type = DataType.NonIID
            ############
            self.epochs_num_input = 2 #20
            self.epochs_num_train = 2 #10
            self.iterations = 2
            self.percent_train_data_use = 0.2
            self.percent_test_relative_to_train = 0.2

        #######-------------------------------------
        #self.update_data_type(data_type)

        #######-------------------------------------
        #self.update_net_type(net_type)

        ##############------------------------------------------------

        if exp_type == ExpType.full:
            self.epochs_num_input = 20
            self.epochs_num_train = 10
            self.iterations = 11
            self.percent_train_data_use = 1
            self.percent_test_relative_to_train = 1












experiment_config = ExperimentConfig()






