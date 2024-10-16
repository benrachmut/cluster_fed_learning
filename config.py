from enum import Enum


class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

iterations = 5
num_clients = 5
percent_train_data_use = 0.05
percent_test_relative_to_train = 0.1
client_split_ratio = 0.8
server_net_type = NetType.VGG
client_net_type = NetType.ALEXNET
num_classes = 10

client_epochs_train = 5
client_batch_size_train = 32
client_learning_rate_train = 0.001

client_epochs_fine_tune = 5
client_batch_size_fine_tune = 32
client_learning_rate_fine_tune = 0.001