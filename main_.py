import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
from functions import *
from entities import *




if __name__ == '__main__':
    print(device)
    torch.manual_seed(seed_num)
    clients_data_dict, server_data_dict, test_set = create_data()
    clients,clients_ids = create_clients(clients_data_dict,server_data_dict,test_set)
    server = Server(id_="server",global_data=server_data_dict,test_data = test_set,clients_ids = clients_ids)

    for t in range(iterations):
        print("----------------------------iter number:"+str(t))
        for c in clients: c.iterate(t)
        for c in clients: server.receive_single_pseudo_label(c.id_,c.pseudo_label_to_send)
        server.iterate(t)
        for c in clients: c.pseudo_label_received = server.pseudo_label_to_send
        file_name= get_file_name(round(server_split_ratio,2))+"_test_avg_loss"
        average_loss_df = create_mean_df(clients,file_name)



        #plot_average_loss(average_loss_df=average_loss_df,filename = file_name)

        # Now compute the average test loss across clients