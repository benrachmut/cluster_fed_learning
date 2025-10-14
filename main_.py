import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.datapipes.dataframe.dataframe_wrapper import iterate
from torchvision.datasets import CIFAR10

import config

from config import *
from functions import *
from entities import *


# record_data.py

class RecordData:
    def __init__(self, clients, server=None):
        self.summary = experiment_config.to_dict()
        self.size_of_client_message = {}

        if server is not None and isinstance(server, Server_Centralized):
            self.server_accuracy_per_cluster = server.accuracy_per_cluster_model
            if experiment_config.algorithm_selection == AlgorithmSelected.MAPL:
                self.server_pseudo_label_before_net = server.pseudo_label_before_net_L2
                self.server_pseudo_label_after_net = server.pseudo_label_after_net_L2
                self.server_global_data_accuracy = server.accuracy_global_data_1
        else:
            if server is not None:
                self.server_accuracy_per_client_1 = server.accuracy_per_client_1
                self.server_accuracy_per_client_1_max = server.accuracy_per_client_1_max
                self.server_accuracy_per_client_10_max = server.accuracy_per_client_10_max
                self.server_accuracy_per_client_100_max = server.accuracy_per_client_100_max
                self.server_accuracy_per_client_5_max = server.accuracy_per_client_5_max

                if experiment_config.algorithm_selection == AlgorithmSelected.MAPL:
                    self.server_pseudo_label_after_net = server.pseudo_label_after_net_L2
                    self.server_pseudo_label_before_net = server.pseudo_label_before_net_L2

        self.client_accuracy_per_client_1 = {}
        self.client_accuracy_per_client_10 = {}
        self.client_accuracy_per_client_100 = {}
        self.client_accuracy_per_client_5 = {}

        self.clients_pseudo_label = {}

        if clients is not None:
            for client in clients:
                id_ = client.id_
                self.client_accuracy_per_client_1[id_] = client.accuracy_per_client_1
                self.client_accuracy_per_client_10[id_] = client.accuracy_per_client_10
                self.client_accuracy_per_client_100[id_] = client.accuracy_per_client_100
                self.client_accuracy_per_client_5[id_] = client.accuracy_per_client_5

                if experiment_config.algorithm_selection == AlgorithmSelected.MAPL:
                    self.clients_pseudo_label[id_] = client.pseudo_label_L2

            if experiment_config.algorithm_selection in (
                AlgorithmSelected.MAPL,
                AlgorithmSelected.FedAvg,
                AlgorithmSelected.pFedCK,
            ):
                for client in clients:
                    id_ = client.id_
                    self.size_of_client_message[id_] = client.size_sent

    # ---- NEW: JSON helpers ----
    def to_dict(self) -> dict:
        """Deep JSON-safe dictionary for this record."""
        return to_json_dict(self)

    def to_json(self, path) -> None:
        """Write JSON file to disk."""
        save_json(self, path)


def clients_and_server_use_pseudo_labels():
    pass



def run_FedAvg():

    for net_type in [NetsType.C_alex_S_alex]:
        experiment_config.update_net_type(net_type)
        for net_cluster_technique in net_cluster_technique_list:
            experiment_config.net_cluster_technique = net_cluster_technique
            for server_input_tech in [ServerInputTech.mean]:
                experiment_config.server_input_tech = server_input_tech
                for cluster_technique in [ClusterTechnique.kmeans]:
                    experiment_config.cluster_technique = cluster_technique
                    for server_feedback_technique in server_feedback_technique_list:
                        experiment_config.server_feedback_technique = server_feedback_technique
                        for num_cluster in num_cluster_list_fedAVG:
                            experiment_config.num_clusters = num_cluster



                            clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                                           server_train_data,
                                                                                           clients_test_data_dict,
                                                                                           server_test_data)
                            server = ServerFedAvg(id_="server", global_data=server_train_data, test_data=server_test_data,
                                        clients_ids=clients_ids, clients_test_data_dict=clients_test_by_id_dict)

                            for t in range(experiment_config.iterations):
                                    print("----------------------------iter number:" + str(t))
                                    for c in clients: c.iterate(t)
                                    for c in clients: server.received_weights[c.id_] = c.weights_to_send
                                    server.iterate(t)
                                    for c in clients: c.weights_received = server.weights_to_send[c.id_]
                                    rd = RecordData(clients, server)


                                    save_record_to_results(rd)



def iterate_fl_clusters(clients,server,net_type,net_cluster_technique,server_input_tech,cluster_technique,server_feedback_technique,
                        num_cluster,weights_for_ps=None,input_consistency=None,epsilon =None):
    for t in range(experiment_config.iterations):
        print("----------------------------iter number:" + str(t))
        for c in clients: c.iterate(t)

        for c in clients:
            what_to_send = c.pseudo_label_to_send

            server.receive_single_pseudo_label(c.id_, what_to_send)
        server.iterate(t)
        for c in clients: c.pseudo_label_received = server.pseudo_label_to_send[c.id_]
        rd = RecordData(clients, server)






        save_record_to_results(rd)



def run_PseudoLabelsClusters():

    for net_type in nets_types_list_PseudoLabelsClusters:
        experiment_config.update_net_type(net_type)
        #if net_type == NetsType.C_alex_S_vgg:
        #    experiment_config.batch_size=128


        for net_cluster_technique in net_cluster_technique_list:
            experiment_config.net_cluster_technique = net_cluster_technique



            for server_input_tech in server_input_tech_list:
                experiment_config.server_input_tech = server_input_tech


                for cluster_technique in cluster_technique_list:
                    experiment_config.cluster_technique = cluster_technique


                    for server_feedback_technique in server_feedback_technique_list:
                        experiment_config.server_feedback_technique = server_feedback_technique

                        experiment_config.num_clusters = -1


                        for epsilon in cluster_additions:
                            experiment_config.cluster_addition = epsilon



                            for weights_for_ps in weights_for_ps_list:
                                experiment_config.weights_for_ps = weights_for_ps

                                for input_consistency in input_consistency_list:
                                    experiment_config.input_consistency = input_consistency


                                    clients, clients_ids, clients_test_by_id_dict = create_clients(
                                        clients_train_data_dict,
                                        server_train_data,
                                        clients_test_data_dict,
                                        server_test_data)
                                    server = Server(id_="server", global_data=server_train_data,
                                                    test_data=server_test_data,
                                                    clients_ids=clients_ids,
                                                    clients_test_data_dict=clients_test_by_id_dict)


                                    iterate_fl_clusters(clients, server, net_type, net_cluster_technique, server_input_tech,
                                                        cluster_technique, server_feedback_technique,
                                                        -1, weights_for_ps,input_consistency,epsilon)



def run_NoFederatedLearning():

    for net_type in [NetsType.C_alex_S_alex]:
        experiment_config.update_net_type(net_type)


        clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                       server_train_data,
                                                                       clients_test_data_dict,
                                                                       server_test_data)
        for c in clients:
            c.fine_tune()
        rd = RecordData(clients)

        save_record_to_results(rd)


def run_Centralized():
    experiment_config.cluster_technique = ClusterTechnique.kmeans
    for net_type in nets_types_Centralized_list:
        experiment_config.which_net_arch = net_type
        experiment_config.update_net_type(net_type)

        for net_cluster_technique in net_cluster_technique_Centralized_list:
            experiment_config.net_cluster_technique = net_cluster_technique


            for cluster_num in num_cluster_Centralized_list:
                experiment_config.num_clusters =cluster_num


                train_data = clients_train_data_dict
                test_data = clients_test_data_dict


                server = Server_Centralized(id_="server", train_data=train_data, test_data=test_data,
                                       evaluate_every=experiment_config.epochs_num_input_fine_tune_clients)

                server.iterate(0)
                rd = RecordData(clients=None, server=server)
                save_record_to_results(rd)



def run_pFedCK():
    for net_type in [NetsType.C_alex_S_vgg]:
        experiment_config.update_net_type(net_type)


        clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                       server_train_data,
                                                                       clients_test_data_dict,
                                                                       server_test_data)
        server = Server_pFedCK(id_="server", global_data=server_train_data, test_data=server_test_data,
                        clients_ids=clients_ids, clients_test_data_dict=clients_test_by_id_dict,clients = clients)

        for t in range(experiment_config.iterations):
            print("----------------------------iter number:" + str(t))

            delta_ws = []

            for client in clients:
                delta_w = client.train(t)  # Handles both models and returns delta_w
                total_size = 0
                for param in delta_w.values():
                    total_size += param.numel() * param.element_size()
                client.size_sent[t] = total_size / (1024 * 1024)



                delta_ws.append(delta_w)

            # Step 2–3: Server clusters and aggregates
            server.cluster_and_aggregate(delta_ws)



            rd = RecordData(clients, server)
            save_record_to_results(rd)


def run_PseudoLabelsNoServerModel():
    for net_type in nets_types_list_PseudoLabelsClusters:
        experiment_config.update_net_type(net_type)


        for net_cluster_technique in net_cluster_technique_list:
            experiment_config.net_cluster_technique = net_cluster_technique

            for server_input_tech in server_input_tech_list:
                experiment_config.server_input_tech = server_input_tech

                for cluster_technique in cluster_technique_list:
                    experiment_config.cluster_technique = cluster_technique

                    for server_feedback_technique in [ServerFeedbackTechnique.similar_to_client]:
                        experiment_config.server_feedback_technique = server_feedback_technique

                        for num_clusters in [1]:
                            experiment_config.num_clusters = num_clusters



                            clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                                           server_train_data,
                                                                                           clients_test_data_dict,
                                                                                           server_test_data)
                            server = Server_PseudoLabelsNoServerModel(id_="server", global_data=server_train_data,
                                                                      test_data=server_test_data,
                                                                      clients_ids=clients_ids,
                                                                      clients_test_data_dict=clients_test_by_id_dict)

                            iterate_fl_clusters(clients, server, net_type, net_cluster_technique, server_input_tech,
                                               cluster_technique, server_feedback_technique,
                                               num_clusters)

def run_COMET():
    for net_type in nets_types_list_PseudoLabelsClusters:
        experiment_config.update_net_type(net_type)

        for net_cluster_technique in net_cluster_technique_list:
            experiment_config.net_cluster_technique = net_cluster_technique

            for server_input_tech in server_input_tech_list:
                experiment_config.server_input_tech = server_input_tech

                for cluster_technique in [ClusterTechnique.kmeans]:
                    experiment_config.cluster_technique = cluster_technique

                    for server_feedback_technique in [ServerFeedbackTechnique.similar_to_client]:
                        experiment_config.server_feedback_technique = server_feedback_technique

                        for num_clusters in [5]:
                            experiment_config.num_clusters = num_clusters



                            clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                                           server_train_data,
                                                                                           clients_test_data_dict,
                                                                                           server_test_data)
                            server = Server_PseudoLabelsNoServerModel(id_="server", global_data=server_train_data,
                                                                      test_data=server_test_data,
                                                                      clients_ids=clients_ids,
                                                                      clients_test_data_dict=clients_test_by_id_dict)

                            iterate_fl_clusters(clients, server, net_type, net_cluster_technique, server_input_tech,
                                               cluster_technique, server_feedback_technique,
                                               num_clusters)

def run_exp_by_algo():
    experiment_config.weights_for_ps = WeightForPS.withoutWeights
    experiment_config.input_consistency = InputConsistency.withoutInputConsistency

    if algorithm_selection == AlgorithmSelected.MAPL :
        run_PseudoLabelsClusters() # Done

    if algorithm_selection == AlgorithmSelected.FedMD:
        run_PseudoLabelsNoServerModel()  # Running

    if algorithm_selection == AlgorithmSelected.COMET:
        run_COMET()

    if algorithm_selection == AlgorithmSelected.NoFederatedLearning:
        run_NoFederatedLearning()

    if algorithm_selection == AlgorithmSelected.Centralized:
        run_Centralized()

    if algorithm_selection == AlgorithmSelected.FedAvg:
        run_FedAvg()

    if algorithm_selection == AlgorithmSelected.pFedCK:
        run_pFedCK()




if __name__ == '__main__':
    print(device)
    seed_num_list = [1]#[2,4,5]#10:[2,4,5,6,9]#100:[1,2,3,5,7]#[1,2,3,4,5,6,7,8,9]
    data_sets_list = [DataSet.CIFAR100]
    num_clients_list = [25]#[25]
    num_opt_clusters_list =[5] #[5]
    mix_percentage = 0.1
    server_split_ratio_list = [0.2]
    alpha_dichts =[5] #[3,2,100,10,5,1] #[3,2,1,]
    cluster_additions = [0]
    server_data_ratios = [1]#[-4,-3,-2,-1,0,1,2,3,4] #  # 0.96,0.5,0.75,1,1.25,1.5,1.75,2]
    print("epsilons:", cluster_additions)
    print(("alpha_dichts", alpha_dichts))
    algorithm_selection_list =[AlgorithmSelected.MAPL]#,AlgorithmSelected.pFedCK,AlgorithmSelected.pFedCK,AlgorithmSelected.COMET,AlgorithmSelected.FedMD]
    #AlgorithmSelected.FedAvg,AlgorithmSelected.NoFederatedLearning,AlgorithmSelected.pFedCK
    #AlgorithmSelected.PseudoLabelsClusters
    #AlgorithmSelected.COMET,AlgorithmSelected.PseudoLabelsNoServerModel

    # parameters for PseudoLabelsClusters
    nets_types_list_PseudoLabelsClusters  = [NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg,NetsType.C_alex_S_alex]#,NetsType.C_alex_S_vgg]# ,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]
    net_cluster_technique_list = [NetClusterTechnique.multi_model]#,NetClusterTechnique.multi_head]
    server_input_tech_list = [ServerInputTech.max]
    cluster_technique_list = [ClusterTechnique.greedy_elimination_L2]#[ClusterTechnique.greedy_elimination_cross_entropy]#[ClusterTechnique.manual_single_iter,ClusterTechnique.manual,ClusterTechnique.kmeans]
    server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster]#[ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]
    #num_cluster_list = [1]#[1,"Optimal"]
    weights_for_ps_list = [WeightForPS.withWeights]#,WeightForPS.withoutWeights ]
    input_consistency_list = [InputConsistency.withInputConsistency]#,InputConsistency.withoutInputConsistency]
    # centralized
    nets_types_Centralized_list = [NetsType.S_vgg]
    num_cluster_Centralized_list = [1]
    net_cluster_technique_Centralized_list = [NetClusterTechnique.multi_model]#,NetClusterTechnique.multi_head]

    #NoFederatedLearning
    nets_types_list_NoFederatedLearning  = [NetsType.C_alex_S_alex]#,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]



    # parameters for fedAvg
    num_cluster_list_fedAVG = [1]
    nets_types_list_fedAVG  = [NetsType.C_alex_S_alex] # dont touch
    cluster_technique_list_fedAVG = [ClusterTechnique.kmeans] # we need this because of logic in num_cluster_list_fedAVG



    ##### create Data #######
    for data_set  in  data_sets_list:
        experiment_config.update_num_classes(data_set)
        for num_clients in num_clients_list:
            experiment_config.num_clients = num_clients
            for num_opt_clusters in  num_opt_clusters_list:
                experiment_config.number_of_optimal_clusters = num_opt_clusters
                experiment_config.identical_clients = int(experiment_config.num_clients / experiment_config.number_of_optimal_clusters)

                for server_split_ratio in server_split_ratio_list:
                    experiment_config.server_split_ratio = server_split_ratio
                    experiment_config.mix_percentage = mix_percentage
                    for  server_data_ratio in    server_data_ratios:
                        experiment_config.server_data_ratio = server_data_ratio

                        for alpha_dicht in alpha_dichts:
                            experiment_config.alpha_dich =alpha_dicht
                            for current_seed_num in seed_num_list:
                                experiment_config.seed_num = current_seed_num
                                clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data = create_data()
                                print("current_seed_num",current_seed_num)

                                for algorithm_selection in algorithm_selection_list :
                                    experiment_config.algorithm_selection = algorithm_selection
                                    if algorithm_selection == AlgorithmSelected.FedMD or algorithm_selection == AlgorithmSelected.COMET:
                                        nets_types_list_PseudoLabelsClusters = [NetsType.C_alex]
                                        net_cluster_technique_list = [NetClusterTechnique.no_model]
                                        server_input_tech_list = [ServerInputTech.mean]
                                        cluster_technique_list = [ClusterTechnique.kmeans]
                                        server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster,
                                                                          ServerFeedbackTechnique.similar_to_client]  # [ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]

                                    torch.manual_seed(experiment_config.seed_num)

                                    run_exp_by_algo()