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
        self.server_accuracy_per_client_1_max = server.accuracy_per_client_1_max
        self.client_accuracy_per_client_1 = {}
        for client in clients:
            id_ = client.id_
            self.client_accuracy_per_client_1[id_]=client.accuracy_per_client_1


if __name__ == '__main__':
    print(device)

    data_sets = [DataSet.CIFAR100,DataSet.CIFAR10]
    num_clients_list = [50]
    mix_percentage_list = [0.2]



    num_cluster_list = ["Optimal", 10, 5, 1]
    nets_types_list  = [NetsType.C_alex_S_alex]#,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]
    cluster_technique_list = [ClusterTechnique.kmeans]
    #server_input_tech_list = [ServerInputTech.max]
    #server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster]#[ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]
    #net_cluster_technique_list = [NetClusterTechnique.multi_model]#,NetClusterTechnique.multi_head]

    torch.manual_seed(experiment_config.seed_num)

    for data_set  in data_sets:
        experiment_config.data_set_selected = data_set
        data_to_pickle = {data_set.name: {}}
        if data_set == DataSet.CIFAR100:
            experiment_config.num_classes = 100
        if data_set == DataSet.CIFAR10:
            experiment_config.num_classes = 10

            for num_clients in num_clients_list:

                experiment_config.num_clients = num_clients
                experiment_config.identical_clients = int(experiment_config.num_clients / experiment_config.number_of_optimal_clusters)

                #experiment_config.identical_clients = int(num_clients/5)
                data_to_pickle[data_set.name][num_clients] = {}
                for mix_percentage in mix_percentage_list:
                    mix_percentage_name = str(int(mix_percentage*100))

                    experiment_config.mix_percentage = mix_percentage
                    data_to_pickle[data_set.name][num_clients][mix_percentage_name] = {}
                    clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data = create_data()

                    for net_type in nets_types_list:
                        experiment_config.update_net_type(net_type)
                        data_to_pickle[data_set.name][num_clients][mix_percentage_name][net_type.name] = {}

                        clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,server_train_data,clients_test_data_dict, server_test_data)
                        server = Server(id_="server", global_data=server_train_data, test_data=server_test_data, clients_ids=clients_ids, clients_test_data_dict=clients_test_by_id_dict)

                        data_to_pickle[data_set.name][num_clients][mix_percentage_name][net_type.name]={}
                        for cluster_technique in cluster_technique_list:
                            experiment_config.cluster_technique = cluster_technique
                            data_to_pickle[data_set.name][num_clients][mix_percentage_name][net_type.name][cluster_technique.name] = {}
                            for num_cluster in num_cluster_list:
                                experiment_config.num_clusters = num_cluster
                                data_to_pickle[data_set.name][num_clients] [mix_percentage_name][net_type.name][cluster_technique.name][num_cluster] = {}
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
                                    data_to_pickle[data_set.name][num_clients] [mix_percentage_name][net_type.name][cluster_technique.name][num_cluster] = rd
                                    pik_name = data_set.name+"_"+str(num_clients)+"_"+mix_percentage_name+"_"+net_type.name+"_"+cluster_technique.name+"_"+str(num_cluster)
                                    #pik_name = net_cluster_technique.name +"_" +net_type.name+"_" + str(num_cluster)+"_" +server_input_tech.name+"_" +cluster_technique.name+"_" +server_feedback_technique.name
                                    pickle_file_path = pik_name +".pkl"

                                    with open(pickle_file_path, "wb") as file:
                                        pickle.dump(data_to_pickle, file)
