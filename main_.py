import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
from functions import *
from entities import *




if __name__ == '__main__':
    train_set, test_set = create_data()
    client_data_sets,server_data = split_clients_server_data(train_set)
    clients,clients_ids = create_clients(client_data_sets,server_data,test_set)
    server = Server(server_data,clients_ids)

    for t in range(iterations):

        for c in clients: c.iterate(t)
        for c in clients: server.receive_single_pseudo_label(c.id_,c.pseudo_label_to_send)
        server.iterate()
        for c in clients: c.pseudo_label_received = server.pseudo_label_to_send


