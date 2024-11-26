from enum import Enum

from matplotlib import pyplot as plt


class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

class DataSet(Enum):
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"


data_set_selected = DataSet.CIFAR10
seed_num = 1
with_server_net = True
epochs_num_input = 10
iterations = 20
server_split_ratio = 0.2

#----------------

num_superclass = 1
num_classes_per_superclass = 2
identical_clients = 2
num_clients = num_superclass*num_classes_per_superclass*identical_clients
#----------------




percent_train_data_use = 1
percent_test_relative_to_train = 1
server_net_type = NetType.VGG
client_net_type = NetType.ALEXNET


#epochs_num_input_train_client = 10
#server_epochs_num_input = 10

client_batch_size_train = 32
client_learning_rate_train = 0.001

client_batch_size_fine_tune = 32
client_learning_rate_fine_tune = 0.001

client_batch_size_evaluate = 32

server_batch_size_train = 32
server_learning_rate_train = 0.0001

server_batch_size_evaluate = 32


def get_CIFAR10_superclass_dict():
    dict_ = {"animal":['bird', 'cat', 'deer', 'dog', 'frog', 'horse'],
             "vehicle":['airplane', 'automobile','ship', 'truck']
             }
    return dict_

def get_meta_data():
    ans = {
        'c_amount':[num_clients],
        'seed':[seed_num],
        'server_data': [server_split_ratio],
        'is_server_net': [with_server_net],  # You might need to pass or save client_split_ratio
        'is_prev_weights': [with_prev_weights],
        'epochs': [epochs_num_input],
        'percent_train_data': [percent_train_data_use]
    }
    return ans

def get_meta_data_text_keys():
    ans = []
    for k in get_meta_data().keys():
        ans.append(k)
    return ans

def file_name():
    ans = ""
    for k,v in get_meta_data().items():
        ans = ans+k+"_"+str(v[0])+"__"
    return ans


