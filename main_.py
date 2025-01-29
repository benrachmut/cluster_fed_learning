import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

import config

from config import *
from functions import *
from entities import *


class RecordData:
    def __init__(self, clients, server):#loss_measures,accuracy_measures,accuracy_pl_measures,accuracy_measures_k,accuracy_pl_measures_k):
        self.summary = experiment_config.to_dict()
        self.server_accuracy_per_client_1 = server.accuracy_per_client_1
        self.server_accuracy_per_client_5 = server.accuracy_per_client_5
        self.client_accuracy_per_client_1 = {}
        self.client_accuracy_per_client_5 = {}

        for client in clients:
            id_ = client.id_
            self.client_accuracy_per_client_1[id_]=client.accuracy_per_client_1
            self.client_accuracy_per_client_5[id_]=client.accuracy_per_client_5


if __name__ == '__main__':
    print(device)




    exp_type = ExpType.full


    data_types =[DataType.NonIID,DataType.IID]

    nets_types_list  = [NetsType.C_alex_S_vgg]
    cluster_technique_list = [ClusterTechnique.manual]
    server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]
    num_cluster_list = [1,5,2,3,4]

    experiment_config.net_cluster_technique= NetClusterTechnique.multi_head

    for data_type in data_types:
        data_to_pickle = {}
        experiment_config.update_data_type(data_type)
        clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data = create_data(data_type)

        for num_cluster in num_cluster_list:

            experiment_config.num_clusters = num_cluster
            if num_cluster == 1:
                experiment_config.num_rounds_multi_head = 1
            else:
                experiment_config.num_rounds_multi_head = 2

            data_to_pickle[num_cluster] = {}
            for net_type in nets_types_list:
                experiment_config.update_net_type(net_type)
                data_to_pickle[num_cluster][net_type.name] = {}
                for cluster_technique in cluster_technique_list:
                    experiment_config.cluster_technique = cluster_technique
                    data_to_pickle[num_cluster][net_type.name][cluster_technique.name] = {}
                    for server_feedback_technique in server_feedback_technique_list:
                        experiment_config.server_feedback_technique = server_feedback_technique
                        experiment_config.update_type_of_experiment(exp_type)
                        torch.manual_seed(experiment_config.seed_num)
                        clients,clients_ids = create_clients(clients_train_data_dict, server_train_data,clients_test_data_dict)
                        server = Server(id_="server",global_data=server_train_data,test_data = server_test_data,clients_ids = clients_ids,clients_test_data_dict = clients_test_data_dict)

                        for t in range(experiment_config.iterations):
                            print("----------------------------iter number:"+str(t))
                            for c in clients:
                                c.iterate(t)
                                print()
                            for c in clients:
                                server.receive_single_pseudo_label(c.id_,c.pseudo_label_to_send)
                            server.iterate(t)
                            for c in clients:
                                c.pseudo_label_received = server.pseudo_label_to_send[c.id_]
                            rd = RecordData(clients, server)
                            data_to_pickle[num_cluster][net_type.name][cluster_technique.name][server_feedback_technique.name]  = rd
                            pik_name = net_type.name+data_type.name+ exp_type.name+"_"+str(num_cluster)+"_"+cluster_technique.name+"_"+server_feedback_technique.name
                            pickle_file_path = pik_name +".pkl"

                            with open(pickle_file_path, "wb") as file:
                                pickle.dump(data_to_pickle, file)
