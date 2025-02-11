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
        self.server_accuracy_per_client_1_max = server.accuracy_per_client_1_max
        self.server_accuracy_server_test_max = server.accuracy_per_client_1
        self.server_accuracy_server_global_max = server.accuracy_global_max
        self.server_accuracy_per_cluster_test_1=server.accuracy_server_test_1
        self.server_accuracy_per_cluster_global_data_1 = server.accuracy_global_data_1
        self.client_accuracy_per_client_1 = {}
        self.client_accuracy_per_client_5 = {}
        self.client_accuracy_test_global_data = {}
        self.client_accuracy_global_data = {}

        for client in clients:
            id_ = client.id_
            self.client_accuracy_per_client_1[id_]=client.accuracy_per_client_1
            self.client_accuracy_per_client_5[id_]=client.accuracy_per_client_5
            self.client_accuracy_test_global_data[id_]=client.accuracy_test_global_data
            self.client_accuracy_global_data[id_]=client.accuracy_global_data

if __name__ == '__main__':
    print(device)




    exp_type = ExpType.full


    data_types =[DataType.NonIID]
    mix_percentage_list = [0.2]
    server_input_tech_list = [ServerInputTech.max]
    nets_types_list  = [NetsType.C_alex_S_alex]#,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]
    cluster_technique_list = [ClusterTechnique.manual]
    server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster]#[ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]

    net_cluster_technique_list = [NetClusterTechnique.multi_head,NetClusterTechnique.multi_model]#,NetClusterTechnique.multi_head]

    for data_type in data_types:
        data_to_pickle ={data_type.name:{}}
        if data_type == DataType.NonIID:
            num_cluster_list = [5,3,"known_labels", 1]
        else:
            num_cluster_list = [1, 5, 2,3,4]
            mix_percentage_list = [1]
        experiment_config.update_data_type(data_type)
        experiment_config.update_type_of_experiment(exp_type)
        for mix_percentage in mix_percentage_list:
            mix_percentage_name = str(int(mix_percentage*100))

            experiment_config.mix_percentage = mix_percentage
            data_to_pickle[data_type.name][mix_percentage_name] = {}
            clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data = create_data(data_type)
            #if experiment_config.percent_train_data_use<1:
            #    clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data = cut_data_v2(clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data)

            for net_type in nets_types_list:
                experiment_config.update_net_type(net_type)
                data_to_pickle[data_type.name][mix_percentage_name][net_type.name] = {}

                for net_cluster_technique in net_cluster_technique_list:
                    experiment_config.net_cluster_technique = net_cluster_technique
                    data_to_pickle[data_type.name][mix_percentage_name][net_type.name][net_cluster_technique.name]={}

                    for num_cluster in num_cluster_list:
                        if num_cluster == "known_labels" and data_type == DataType.IID:
                            continue
                        else:
                            experiment_config.num_clusters = num_cluster
                            if num_cluster == 1:
                                experiment_config.num_rounds_multi_head = 1
                            else:
                                experiment_config.num_rounds_multi_head = 2
                            data_to_pickle[data_type.name][mix_percentage_name][net_type.name][net_cluster_technique.name][num_cluster] = {}






                        for server_input_tech in server_input_tech_list:
                            experiment_config.server_input_tech = server_input_tech
                            data_to_pickle[data_type.name][mix_percentage_name][net_type.name][net_cluster_technique.name][num_cluster][server_input_tech.name] = {}

                            for cluster_technique in cluster_technique_list:
                                experiment_config.cluster_technique = cluster_technique
                                data_to_pickle[data_type.name][mix_percentage_name][net_type.name][net_cluster_technique.name][num_cluster][server_input_tech.name][cluster_technique.name] = {}


                                for server_feedback_technique in server_feedback_technique_list:
                                    experiment_config.server_feedback_technique = server_feedback_technique
                                    experiment_config.update_type_of_experiment(exp_type)
                                    torch.manual_seed(experiment_config.seed_num)
                                    clients,clients_ids = create_clients(clients_train_data_dict, server_train_data,clients_test_data_dict,server_test_data)
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
                                        data_to_pickle[data_type.name][mix_percentage_name][net_type.name][net_cluster_technique.name][num_cluster][server_input_tech.name][
                                            cluster_technique.name][server_feedback_technique.name] = rd
                                        pik_name = data_type.name+"_"+mix_percentage_name+"_"+net_type.name+"_"+str(num_cluster)+"_"+net_cluster_technique.name+"_"+server_input_tech.name+"_"+cluster_technique.name +"_"+server_feedback_technique.name
                                        #pik_name = net_cluster_technique.name +"_" +net_type.name+"_" + str(num_cluster)+"_" +server_input_tech.name+"_" +cluster_technique.name+"_" +server_feedback_technique.name
                                        pickle_file_path = pik_name +".pkl"

                                        with open(pickle_file_path, "wb") as file:
                                            pickle.dump(data_to_pickle, file)
