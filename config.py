from enum import Enum


class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

iterations = 2
num_clients = 2
percent_train_data_use = 0.2
percent_test_relative_to_train = 0.2
client_split_ratio_list = [0.8,0.5,0.2,0.1]
server_net_type = NetType.VGG
client_net_type = NetType.ALEXNET
num_classes = 10

client_epochs_train = 10
client_batch_size_train = 50
client_learning_rate_train = 0.001

client_epochs_fine_tune = 10
client_batch_size_fine_tune = 32
client_learning_rate_fine_tune = 0.001

client_batch_size_evaluate = 32

server_epochs_train = 10
server_batch_size_train = 64
server_learning_rate_train = 0.001

server_batch_size_evaluate = 64