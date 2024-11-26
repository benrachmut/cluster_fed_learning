from enum import Enum

from matplotlib import pyplot as plt


class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"
client_split_ratio = 0.2
seed_num = 1
with_server_net = True
epochs_num_input = 50
iterations = 100
num_clients = 2
percent_train_data_use = 1
percent_test_relative_to_train = 1
server_split_ratio = 0.2
server_net_type = NetType.VGG
client_net_type = NetType.ALEXNET
num_classes = 10


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


