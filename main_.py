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
    def __init__(self, clients, server=None):#loss_measures,accuracy_measures,accuracy_pl_measures,accuracy_measures_k,accuracy_pl_measures_k):
        self.summary = experiment_config.to_dict()
        if server is not None and isinstance(server,Server_Centralized):
            self.server_accuracy_per_cluster = server.accuracy_per_cluster_model


        else:
            if server is not None:
                self.server_accuracy_per_client_1 = server.accuracy_per_client_1
                self.server_accuracy_per_client_1_max = server.accuracy_per_client_1_max
            self.client_accuracy_per_client_1 = {}
            for client in clients:
                id_ = client.id_
                self.client_accuracy_per_client_1[id_]=client.accuracy_per_client_1


def clients_and_server_use_pseudo_labels():
    pass


def run_PseudoLabelsClusters():



    for net_type in nets_types_list_PseudoLabelsClusters:
        experiment_config.update_net_type(net_type)
        data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio][
            algorithm_selection.name][net_type.name] = {}


        for net_cluster_technique in net_cluster_technique_list:
            experiment_config.net_cluster_technique = net_cluster_technique
            data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio][
                algorithm_selection.name][net_type.name][net_cluster_technique.name] = {}



            for server_input_tech in server_input_tech_list:
                experiment_config.server_input_tech = server_input_tech
                data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio][
                    algorithm_selection.name][net_type.name][net_cluster_technique.name][server_input_tech.name] = {}

                for cluster_technique in cluster_technique_list:
                    experiment_config.cluster_technique = cluster_technique
                    data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][
                        server_split_ratio][algorithm_selection.name][net_type.name][net_cluster_technique.name][
                        server_input_tech.name][cluster_technique.name] = {}


                    for server_feedback_technique in server_feedback_technique_list:
                        experiment_config.server_feedback_technique = server_feedback_technique
                        data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][
                            server_split_ratio][algorithm_selection.name][net_type.name][net_cluster_technique.name][
                            server_input_tech.name][cluster_technique.name][server_feedback_technique.name] = {}




                        for num_cluster in num_cluster_list:
                            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy:
                                experiment_config.num_clusters = -1
                            else:
                                experiment_config.num_clusters = num_cluster

                            if experiment_config.algorithm_selection == AlgorithmSelected.PseudoLabelsClusters_with_division:
                                server_train_data_ = fix_global_data(server_train_data)
                                clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                                               server_train_data_,
                                                                                               clients_test_data_dict,
                                                                                               server_test_data)

                                server = Server_PseudoLabelsClusters_with_division(id_="server", global_data=server_train_data_, test_data=server_test_data,
                                                clients_ids=clients_ids, clients_test_data_dict=clients_test_by_id_dict)

                            if algorithm_selection == AlgorithmSelected.PseudoLabelsClusters:
                                clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                                           server_train_data,
                                                                                           clients_test_data_dict,
                                                                                           server_test_data)
                                server = Server(id_="server", global_data=server_train_data, test_data=server_test_data,
                                            clients_ids=clients_ids, clients_test_data_dict=clients_test_by_id_dict)

                            if algorithm_selection == AlgorithmSelected.PseudoLabelsNoServerModel:
                                clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                                           server_train_data,
                                                                                           clients_test_data_dict,
                                                                                           server_test_data)
                                server = Server_PseudoLabelsNoServerModel(id_="server", global_data=server_train_data, test_data=server_test_data,
                                            clients_ids=clients_ids, clients_test_data_dict=clients_test_by_id_dict)

                            for t in range(experiment_config.iterations):
                                    print("----------------------------iter number:" + str(t))
                                    for c in clients: c.iterate(t)
                                    for c in clients: server.receive_single_pseudo_label(c.id_, c.pseudo_label_to_send)
                                    server.iterate(t)
                                    for c in clients: c.pseudo_label_received = server.pseudo_label_to_send[c.id_]
                                    rd = RecordData(clients, server)

                                    data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][
                                        server_split_ratio][algorithm_selection.name][net_type.name][ net_cluster_technique.name][
                                        server_input_tech.name][cluster_technique.name][server_feedback_technique.name][
                                        num_cluster]  = rd
                                    pik_name = data_set.name+"_"+str(num_clients)+"_"+str(num_opt_clusters)+"_"+str(int(10*(server_split_ratio)))+"_"+algorithm_selection.name+"_"+net_type.name+"_"+net_cluster_technique.name+"_"+cluster_technique.name+"_"+str(num_cluster)

                                    pickle_file_path = pik_name + ".pkl"

                                    with open(pickle_file_path, "wb") as file:
                                        pickle.dump(data_to_pickle, file)
                            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy or   experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_L2:
                                break

def run_NoFederatedLearning():

    for net_type in nets_types_list_NoFederatedLearning:
        experiment_config.update_net_type(net_type)
        data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio][
            algorithm_selection.name][net_type.name] = {}

        clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                       server_train_data,
                                                                       clients_test_data_dict,
                                                                       server_test_data)
        for c in clients:
            c.fine_tune()
        rd = RecordData(clients)

        data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][
            server_split_ratio][algorithm_selection.name][net_type.name] = rd
        pik_name = data_set.name + "_" + str(num_clients) + "_" + str(num_opt_clusters) + "_" + str(int(10 * (
            server_split_ratio))) + "_" + algorithm_selection.name + "_" + net_type.name

        pickle_file_path = pik_name + ".pkl"

        with open(pickle_file_path, "wb") as file:
            pickle.dump(data_to_pickle, file)


def run_Centralized():

    for net_type in nets_types_Centralized_list:
        experiment_config.which_net_arch = net_type
        experiment_config.update_net_type(net_type)
        data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio][
            algorithm_selection.name][net_type.name] = {}

        for net_cluster_technique in net_cluster_technique_Centralized_list:
            experiment_config.net_cluster_technique = net_cluster_technique
            data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio][
                algorithm_selection.name][net_type.name][net_cluster_technique] = {}#[cluster_num]


            for cluster_num in num_cluster_Centralized_list:
                experiment_config.num_clusters =cluster_num


                train_data = clients_train_data_dict
                test_data = clients_test_data_dict


                server = Server_Centralized(id_="server", train_data=train_data, test_data=test_data,
                                       evaluate_every=experiment_config.epochs_num_input_fine_tune_clients)

                server.iterate(0)
                rd = RecordData(clients=None, server=server)
                data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio][
                    algorithm_selection.name][net_type.name][net_cluster_technique][cluster_num] = rd

                pik_name = data_set.name + "_" + str(num_clients) + "_" + str(num_opt_clusters) + "_" + str(int(10 * (
                    server_split_ratio))) + "_" + algorithm_selection.name + "_" + net_type.name + "_" + net_cluster_technique.name + "_" + str(
                    str(cluster_num))

                pickle_file_path = pik_name + ".pkl"

                with open(pickle_file_path, "wb") as file:
                    pickle.dump(data_to_pickle, file)


if __name__ == '__main__':
    print(device)
    torch.manual_seed(experiment_config.seed_num)

    data_sets_list = [DataSet.CIFAR100]
    num_clients_list = [25]#[25]
    num_opt_clusters_list =[5] #[5]
    mix_percentage_list = [0.2]
    server_split_ratio_list = [0.2]

    algorithm_selection_list = [AlgorithmSelected.PseudoLabelsClusters]

    # centralized
    nets_types_Centralized_list = [NetsType.S_alex,NetsType.S_vgg]
    num_cluster_Centralized_list = [1,"Optimal"]
    net_cluster_technique_Centralized_list = [NetClusterTechnique.multi_model]#,NetClusterTechnique.multi_head]

    #NoFederatedLearning
    nets_types_list_NoFederatedLearning  = [NetsType.C_alex_S_alex]#,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]


    # parameters for PseudoLabelsClusters
    nets_types_list_PseudoLabelsClusters  = [NetsType.C_alex_S_alex]#,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]
    net_cluster_technique_list = [NetClusterTechnique.multi_model]#,NetClusterTechnique.multi_head]
    server_input_tech_list = [ServerInputTech.max]
    cluster_technique_list = [ClusterTechnique.manual_cross_entropy,ClusterTechnique.manual_L2,ClusterTechnique.kmeans]#[ClusterTechnique.manual_single_iter,ClusterTechnique.manual,ClusterTechnique.kmeans]
    server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster]#[ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]
    num_cluster_list = [5]





    ##### create Data #######
    for data_set  in  data_sets_list:
        data_to_pickle = {data_set.name: {}}
        experiment_config.update_num_classes(data_set)
        for num_clients in num_clients_list:
            experiment_config.num_clients = num_clients
            data_to_pickle[data_set.name][num_clients] = {}
            for num_opt_clusters in  num_opt_clusters_list:
                experiment_config.number_of_optimal_clusters = num_opt_clusters
                data_to_pickle[data_set.name][num_clients][num_opt_clusters] = {}
                experiment_config.identical_clients = int(experiment_config.num_clients / experiment_config.number_of_optimal_clusters)
                for mix_percentage in mix_percentage_list:
                    experiment_config.mix_percentage = mix_percentage
                    data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage] = {}
                    for server_split_ratio in server_split_ratio_list:
                        experiment_config.server_split_ratio = server_split_ratio
                        data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio] = {}


                        clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data = create_data()

                        for algorithm_selection in algorithm_selection_list:

                            if algorithm_selection == AlgorithmSelected.PseudoLabelsNoServerModel:
                                nets_types_list_PseudoLabelsClusters = [NetsType.C_alex]
                                net_cluster_technique_list = [NetClusterTechnique.no_model]
                                server_input_tech_list = [ServerInputTech.mean]
                                cluster_technique_list = [ClusterTechnique.kmeans]
                                server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]  # [ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]


                            experiment_config.algorithm_selection = algorithm_selection
                            data_to_pickle[data_set.name][num_clients][num_opt_clusters][mix_percentage][server_split_ratio][algorithm_selection.name] = {}

                            if algorithm_selection ==AlgorithmSelected.PseudoLabelsClusters or algorithm_selection == AlgorithmSelected.PseudoLabelsNoServerModel or algorithm_selection ==AlgorithmSelected.PseudoLabelsClusters_with_division:
                                run_PseudoLabelsClusters()


                            if algorithm_selection ==AlgorithmSelected.NoFederatedLearning:
                                run_NoFederatedLearning()
                            
                            if algorithm_selection == AlgorithmSelected.Centralized:
                                run_Centralized()