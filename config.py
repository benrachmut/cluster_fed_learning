import pickle
from enum import Enum
from random import random

import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset



class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

class DataSet(Enum):
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"


data_set_selected = DataSet.CIFAR10
mix_percentage = 0.2
seed_num = 1

with_server_net = False
epochs_num_input = 2
iterations = 20
server_split_ratio = 0.2
batch_size = 32
learning_rate = 0.001

#----------------

num_classes = 2
identical_clients = 1

num_clients = num_classes*identical_clients
#----------------



num_clusters = 1
percent_train_data_use = 0.05
percent_test_relative_to_train = 1
server_net_type = NetType.VGG
client_net_type = NetType.ALEXNET


summary = (
    f"num_clusters_{num_clusters}_"
    f"Mix_Percentage_{mix_percentage}_"
    f"Seed_Num_{seed_num}_"
    f"With_Server_Net_{with_server_net}_"
    f"Epochs_{epochs_num_input}_"
    f"Iterations_{iterations}_"
    f"Server_Split_Ratio_{server_split_ratio}_"
    f"Num_Classes_{num_classes}_"
    f"Identical_Clients_{identical_clients}"
)

#epochs_num_input_train_client = 10
#server_epochs_num_input = 10


#client_batch_size_train = 32
#client_learning_rate_train = 0.001

#client_batch_size_fine_tune = 32
#client_learning_rate_fine_tune = 0.001

#client_batch_size_evaluate = 32

#server_batch_size_train = 32
#server_learning_rate_train = 0.0001

#server_batch_size_evaluate = 32


#def get_CIFAR10_superclass_dict():
#    dict_ = {"animal":['bird', 'cat', 'deer', 'dog', 'frog', 'horse'],
#             "vehicle":['airplane', 'automobile','ship', 'truck']
#             }
#    return dict_




def transform_to_TensorDataset(data_):
    images = [item[0] for item in data_]  # Extract the image tensors (index 0 of each tuple)
    targets = [item[1] for item in data_]

    # Step 2: Convert the lists of images and targets into tensors (if not already)
    images_tensor = torch.stack(images)  # Stack the image tensors into a single tensor
    targets_tensor = torch.tensor(targets)  # Convert the targets to a tensor


    # Step 3: Create a TensorDataset from the images and targets
    return TensorDataset(images_tensor, targets_tensor)

class RecordData:
    def __init__(self,loss_measures,loss_measures_class_yes,loss_measures_class_no,accuracy_measures,accuracy_measures_class_yes,accuracy_measures_class_no):
        self.loss_measures = loss_measures
        self.loss_measures_class_yes = loss_measures_class_yes
        self.loss_measures_class_no = loss_measures_class_no

        self.accuracy_measures = accuracy_measures
        self.accuracy_measures_class_yes = accuracy_measures_class_yes
        self.accuracy_measures_class_no = accuracy_measures_class_no



        self.mix_percentage = mix_percentage
        self.seed_num= seed_num
        self.epochs_num_input= epochs_num_input
        self.iterations= iterations
        self. server_split_ratio= server_split_ratio
        self.num_classes=num_classes
        self.identical_clients =identical_clients
        self.num_clusters= num_clusters

        self.summary = (
            f"num_clusters_{num_clusters}_"
            f"Mix_Percentage_{mix_percentage}_"
            f"Epochs_{epochs_num_input}_"
            f"Iterations_{iterations}_"
            f"Server_Split_Ratio_{server_split_ratio}_"
            f"Num_Classes_{num_classes}_"
            f"Identical_Clients_{identical_clients}"
        )

