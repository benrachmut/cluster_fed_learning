import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
from functions import *
from entities import *


class RecordData:
    def __init__(self,loss_measures,accuracy_measures,accuracy_pl_measures):
        self.loss_measures = loss_measures
        self.accuracy_pl_measures = accuracy_pl_measures
        self.accuracy_test_measures = accuracy_measures
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

def create_record_data(clients, server):
    loss_measures = {}
    accuracy_test_measures = {}
    accuracy_pl_measures = {}

    for client in clients:
        loss_measures[client.id_]=client.loss_measures
        accuracy_test_measures[client.id_]=client.accuracy_test_measures
        accuracy_pl_measures[client.id_]=client.accuracy_pl_measures
    loss_measures[server.id_] = server.loss_measures
    accuracy_test_measures[server.id_] = server.accuracy_test_measures
    accuracy_pl_measures[server.id_] = server.accuracy_pl_measures

    return RecordData(loss_measures,accuracy_test_measures,accuracy_pl_measures)


def create_pickle(clients, server):
    rd = create_record_data(clients, server)
    pik_name = rd.summary
    pickle_file_path = pik_name+".pkl"

    with open(pickle_file_path, "wb") as file:
        pickle.dump(rd, file)


if __name__ == '__main__':
    print(device)
    torch.manual_seed(seed_num)
    clients_data_dict, server_data, test_set = create_data()
    clients,clients_ids = create_clients(clients_data_dict,server_data,test_set)
    server = Server(id_="server",global_data=server_data,test_data = test_set,clients_ids = clients_ids)

    for t in range(iterations):
        print("----------------------------iter number:"+str(t))
        for c in clients:
            c.iterate(t)
            print()
        for c in clients:
            server.receive_single_pseudo_label(c.id_,c.pseudo_label_to_send)
        server.iterate(t)
        for c in clients:
            c.pseudo_label_received = server.pseudo_label_to_send
        #create_pickle(clients,server)



        #plot_average_loss(average_loss_df=average_loss_df,filename = file_name)

        # Now compute the average test loss across clients