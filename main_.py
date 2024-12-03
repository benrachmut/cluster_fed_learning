import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
from functions import *
from entities import *
from xlsxwriter import *



if __name__ == '__main__':
    print(device)

    torch.manual_seed(1)  # Set seed for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)  # Set seed for all GPUs
    clients_data_dict, server_data_dict, test_set = create_data()
    clients,clients_ids = create_clients(clients_data_dict,server_data_dict,test_set)
    server = None
    #Server(id_ = "server", global_data=server_data, test_set=test_set, server_split_ratio=server_split_ratio, clients_ids=clients_ids)

    for t in range(iterations):
        print("----------------------------iter number:"+str(t))
        for c in clients: c.iterate(t)
        for c in clients: server.receive_single_pseudo_label(c.id_,c.pseudo_label_to_send)
        create_csv(clients,server)
        server.iterate(t)
        for c in clients: c.pseudo_label_for_train = server.pseudo_label_to_send
        create_csv(clients,server)


        #file_name= get_file_name(round(1-client_split_ratio,2))+"_test_avg_loss"
        #average_loss_df = create_mean_df(clients,file_name)
        #plot_average_loss(average_loss_df=average_loss_df,filename = file_name)

        # Now compute the average test loss across clients



