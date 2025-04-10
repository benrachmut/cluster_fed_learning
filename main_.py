import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.datapipes.dataframe.dataframe_wrapper import iterate

import config

from config import *
from functions import *
from entities import *


class RecordData:
    def __init__(self, clients, server=None):#loss_measures,accuracy_measures,accuracy_pl_measures,accuracy_measures_k,accuracy_pl_measures_k):
        self.summary = experiment_config.to_dict()
        self.size_of_client_message = {}

        if server is not None and isinstance(server,Server_Centralized):
            self.server_accuracy_per_cluster = server.accuracy_per_cluster_model
            if experiment_config.algorithm_selection == AlgorithmSelected.PseudoLabelsClusters:
                self.server_pseudo_label_before_net = server.pseudo_label_before_net_L2
                self.server_pseudo_label_after_net = server.pseudo_label_after_net_L2
                self.server_global_data_accuracy = server.accuracy_global_data_1

        else:
            if server is not None:
                self.server_accuracy_per_client_1 = server.accuracy_per_client_1
                self.server_accuracy_per_client_1_max = server.accuracy_per_client_1_max
                if experiment_config.algorithm_selection == AlgorithmSelected.PseudoLabelsClusters:
                    self.server_pseudo_label_after_net = server.pseudo_label_after_net_L2
                    self.server_pseudo_label_before_net = server.pseudo_label_before_net_L2

        self.client_accuracy_per_client_1 = {}
        self.clients_pseudo_label={}

        if clients is not None:
            for client in clients:
                id_ = client.id_
                self.client_accuracy_per_client_1[id_]=client.accuracy_per_client_1
                if experiment_config.algorithm_selection == AlgorithmSelected.PseudoLabelsClusters:
                    self.clients_pseudo_label[id_] = client.pseudo_label_L2

            if experiment_config.algorithm_selection == AlgorithmSelected.PseudoLabelsClusters or experiment_config.algorithm_selection == AlgorithmSelected.FedAvg:
                for client in clients:
                    id_ = client.id_
                    self.size_of_client_message[id_] = client.size_sent


def clients_and_server_use_pseudo_labels():
    pass



def run_FedAvg():

    for net_type in [NetsType.C_alex_S_alex]:
        experiment_config.update_net_type(net_type)
        data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht][
            algorithm_selection.name][net_type.name] = {}


        for net_cluster_technique in net_cluster_technique_list:
            experiment_config.net_cluster_technique = net_cluster_technique
            data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht][
                algorithm_selection.name][net_type.name][net_cluster_technique.name] = {}



            for server_input_tech in server_input_tech_list:
                experiment_config.server_input_tech = server_input_tech
                data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht][
                    algorithm_selection.name][net_type.name][net_cluster_technique.name][server_input_tech.name] = {}

                for cluster_technique in [ClusterTechnique.kmeans]:
                    experiment_config.cluster_technique = cluster_technique
                    data_to_pickle[data_set.name][num_clients][num_opt_clusters][
                        server_split_ratio][algorithm_selection.name][net_type.name][net_cluster_technique.name][
                        server_input_tech.name][cluster_technique.name] = {}


                    for server_feedback_technique in server_feedback_technique_list:
                        experiment_config.server_feedback_technique = server_feedback_technique
                        data_to_pickle[data_set.name][num_clients][num_opt_clusters][
                            server_split_ratio][algorithm_selection.name][net_type.name][net_cluster_technique.name][
                            server_input_tech.name][cluster_technique.name][server_feedback_technique.name] = {}

                        for num_cluster in num_cluster_list_fedAVG:
                            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy:
                                experiment_config.num_clusters = -1
                            else:
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

                                    data_to_pickle[data_set.name][num_clients][num_opt_clusters][
                                        server_split_ratio][algorithm_selection.name][net_type.name][ net_cluster_technique.name][
                                        server_input_tech.name][cluster_technique.name][server_feedback_technique.name][
                                        num_cluster]  = rd
                                    pik_name = data_set.name+"_"+str(num_clients)+"_"+str(num_opt_clusters)+"_"+str(int(10*(server_split_ratio)))+"_"+algorithm_selection.name+"_"+net_type.name+"_"+net_cluster_technique.name+"_"+cluster_technique.name+"_"+str(num_cluster)+"_"+ str(experiment_config.alpha_dich)

                                    pickle_file_path = pik_name + ".pkl"

                                    with open(pickle_file_path, "wb") as file:
                                        pickle.dump(data_to_pickle, file)
                            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy or   experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_L2:
                                break


def iterate_fl_clusters(clients,server,net_type,net_cluster_technique,server_input_tech,cluster_technique,server_feedback_technique,
                        num_cluster,epsilon =None):
    for t in range(experiment_config.iterations):
        print("----------------------------iter number:" + str(t))
        for c in clients: c.iterate(t)

        for c in clients:
            what_to_send = c.pseudo_label_to_send

            server.receive_single_pseudo_label(c.id_, what_to_send)
        server.iterate(t)
        for c in clients: c.pseudo_label_received = server.pseudo_label_to_send[c.id_]
        rd = RecordData(clients, server)
        if epsilon is None:
            data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][
                algorithm_selection.name][net_type.name][net_cluster_technique.name][
                server_input_tech.name][cluster_technique.name][server_feedback_technique.name][server_feedback_technique.name][
                num_cluster] = rd
            pik_name = data_set.name + "_" + str(num_clients) + "_" + str(
                num_opt_clusters) + "_" + str(int(10 * (
                server_split_ratio))) + "_" + algorithm_selection.name + "_" + net_type.name + "_" + net_cluster_technique.name + "_" + cluster_technique.name + "_" + str(
                num_cluster) + "_" + str(experiment_config.alpha_dich)
        else:
            data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][
                algorithm_selection.name][net_type.name][net_cluster_technique.name][
                server_input_tech.name][cluster_technique.name][server_feedback_technique.name][epsilon] = rd
            pik_name = data_set.name + "_" + str(num_clients) + "_" + str(
                num_opt_clusters) + "_" + str(int(10 * (
                server_split_ratio))) + "_" + algorithm_selection.name + "_" + net_type.name + "_" + net_cluster_technique.name + "_" + cluster_technique.name + "_" + str(
                num_cluster) +"_"+ str(experiment_config.alpha_dich)+"_"+ str(epsilon)


        pickle_file_path = pik_name + ".pkl"

        with open(pickle_file_path, "wb") as file:
            pickle.dump(data_to_pickle, file)


def run_PseudoLabelsClusters():

    for net_type in nets_types_list_PseudoLabelsClusters:
        experiment_config.update_net_type(net_type)
        if net_type == NetsType.C_alex_S_vgg:
            experiment_config.batch_size=128
        data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht][algorithm_selection.name][net_type.name] = {}


        for net_cluster_technique in net_cluster_technique_list:
            experiment_config.net_cluster_technique = net_cluster_technique
            data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht][
                algorithm_selection.name][net_type.name][net_cluster_technique.name] = {}



            for server_input_tech in server_input_tech_list:
                experiment_config.server_input_tech = server_input_tech
                data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][
                    algorithm_selection.name][net_type.name][net_cluster_technique.name][server_input_tech.name] = {}

                for cluster_technique in cluster_technique_list:
                    experiment_config.cluster_technique = cluster_technique
                    data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][algorithm_selection.name][net_type.name][net_cluster_technique.name][
                        server_input_tech.name][cluster_technique.name] = {}


                    for server_feedback_technique in server_feedback_technique_list:
                        experiment_config.server_feedback_technique = server_feedback_technique
                        data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][algorithm_selection.name][net_type.name][net_cluster_technique.name][
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

                            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy or experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_L2:
                                for epsilon in epsilons:
                                    experiment_config.epsilon = epsilon

                                    iterate_fl_clusters(clients,server,net_type,net_cluster_technique,server_input_tech,cluster_technique,server_feedback_technique,
                        num_cluster,epsilon)
                            else:
                                iterate_fl_clusters(clients, server, net_type, net_cluster_technique, server_input_tech,
                                                    cluster_technique, server_feedback_technique,
                                                    num_cluster)

                            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy or   experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_L2:
                                break

def run_NoFederatedLearning():

    for net_type in nets_types_list_NoFederatedLearning:
        experiment_config.update_net_type(net_type)
        data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][
            algorithm_selection.name][net_type.name] = {}

        clients, clients_ids, clients_test_by_id_dict = create_clients(clients_train_data_dict,
                                                                       server_train_data,
                                                                       clients_test_data_dict,
                                                                       server_test_data)
        for c in clients:
            c.fine_tune()
        rd = RecordData(clients)

        data_to_pickle[data_set.name][num_clients][num_opt_clusters][
            server_split_ratio][algorithm_selection.name][net_type.name] = rd
        pik_name = data_set.name + "_" + str(num_clients) + "_" + str(num_opt_clusters) + "_" + str(int(10 * (
            server_split_ratio))) + "_" + algorithm_selection.name + "_" + net_type.name+"_"+ str(experiment_config.alpha_dich)

        pickle_file_path = pik_name + ".pkl"

        with open(pickle_file_path, "wb") as file:
            pickle.dump(data_to_pickle, file)


def run_Centralized():
    experiment_config.cluster_technique = ClusterTechnique.kmeans
    for net_type in nets_types_Centralized_list:
        experiment_config.which_net_arch = net_type
        experiment_config.update_net_type(net_type)
        data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][
            algorithm_selection.name][net_type.name] = {}

        for net_cluster_technique in net_cluster_technique_Centralized_list:
            experiment_config.net_cluster_technique = net_cluster_technique
            data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht][
                algorithm_selection.name][net_type.name][net_cluster_technique] = {}#[cluster_num]


            for cluster_num in num_cluster_Centralized_list:
                experiment_config.num_clusters =cluster_num


                train_data = clients_train_data_dict
                test_data = clients_test_data_dict


                server = Server_Centralized(id_="server", train_data=train_data, test_data=test_data,
                                       evaluate_every=experiment_config.epochs_num_input_fine_tune_clients)

                server.iterate(0)
                rd = RecordData(clients=None, server=server)
                data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht][
                    algorithm_selection.name][net_type.name][net_cluster_technique][cluster_num] = rd

                pik_name = data_set.name + "_" + str(num_clients) + "_" + str(num_opt_clusters) + "_" + str(int(10 * (
                    server_split_ratio))) + "_" + algorithm_selection.name + "_" + net_type.name + "_" + net_cluster_technique.name + "_" + str(
                    str(cluster_num))+"_"+ str(experiment_config.alpha_dich)

                pickle_file_path = pik_name + ".pkl"

                with open(pickle_file_path, "wb") as file:
                    pickle.dump(data_to_pickle, file)


def run_pFedCK():
    for net_type in nets_types_list_PseudoLabelsClusters:
        experiment_config.update_net_type(net_type)
        if net_type == NetsType.C_alex_S_vgg:
            experiment_config.batch_size = 128
        data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
            alpha_dicht][algorithm_selection.name][net_type.name] = {}

        for net_cluster_technique in net_cluster_technique_list:
            experiment_config.net_cluster_technique = net_cluster_technique
            data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                alpha_dicht][
                algorithm_selection.name][net_type.name][net_cluster_technique.name] = {}

            for server_input_tech in server_input_tech_list:
                experiment_config.server_input_tech = server_input_tech
                data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][
                    algorithm_selection.name][net_type.name][net_cluster_technique.name][server_input_tech.name] = {}

                for cluster_technique in cluster_technique_list:
                    experiment_config.cluster_technique = cluster_technique
                    data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][
                        algorithm_selection.name][net_type.name][net_cluster_technique.name][
                        server_input_tech.name][cluster_technique.name] = {}

                    for server_feedback_technique in server_feedback_technique_list:
                        experiment_config.server_feedback_technique = server_feedback_technique
                        data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][alpha_dicht][
                            algorithm_selection.name][net_type.name][net_cluster_technique.name][
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

                                server = Server_PseudoLabelsClusters_with_division(id_="server",
                                                                                   global_data=server_train_data_,
                                                                                   test_data=server_test_data,
                                                                                   clients_ids=clients_ids,
                                                                                   clients_test_data_dict=clients_test_by_id_dict)

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
                                server = Server_PseudoLabelsNoServerModel(id_="server", global_data=server_train_data,
                                                                          test_data=server_test_data,
                                                                          clients_ids=clients_ids,
                                                                          clients_test_data_dict=clients_test_by_id_dict)

                            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy or experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_L2:
                                for epsilon in epsilons:
                                    experiment_config.epsilon = epsilon

                                    iterate_fl_clusters(clients, server, net_type, net_cluster_technique,
                                                        server_input_tech, cluster_technique, server_feedback_technique,
                                                        num_cluster, epsilon)
                            else:
                                iterate_fl_clusters(clients, server, net_type, net_cluster_technique, server_input_tech,
                                                    cluster_technique, server_feedback_technique,
                                                    num_cluster)

                            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy or experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_L2:
                                break






def run_exp_by_algo():
    if algorithm_selection == AlgorithmSelected.PseudoLabelsClusters or algorithm_selection == AlgorithmSelected.PseudoLabelsNoServerModel or algorithm_selection == AlgorithmSelected.PseudoLabelsClusters_with_division:
        run_PseudoLabelsClusters()

    if algorithm_selection == AlgorithmSelected.NoFederatedLearning:
        run_NoFederatedLearning()

    if algorithm_selection == AlgorithmSelected.Centralized:
        run_Centralized()

    if algorithm_selection == AlgorithmSelected.FedAvg:
        run_FedAvg()

    #if algorithm_selection == AlgorithmSelected.pFedCK:
        #run_pFedCK()




if __name__ == '__main__':
    print(device)
    torch.manual_seed(experiment_config.seed_num)

    data_sets_list = [DataSet.CIFAR100]
    num_clients_list = [25]#[25]
    num_opt_clusters_list =[5] #[5]
    mix_percentage = 0.2
    server_split_ratio_list = [0.2]
    alpha_dichts = [5]
    epsilons = [1]  # 0.96,0.5,0.75,1,1.25,1.5,1.75,2]
    print("epsilons:", epsilons)
    print(("alpha_dichts", alpha_dichts))
    algorithm_selection_list = [ AlgorithmSelected.PseudoLabelsClusters]#[ AlgorithmSelected.PseudoLabelsClusters,AlgorithmSelected.PseudoLabelsNoServerModel,AlgorithmSelected.FedAvg,AlgorithmSelected.Centralized,AlgorithmSelected.NoFederatedLearning]

    # centralized
    nets_types_Centralized_list = [NetsType.S_alex,NetsType.S_vgg]
    num_cluster_Centralized_list = [1,"Optimal"]
    net_cluster_technique_Centralized_list = [NetClusterTechnique.multi_model]#,NetClusterTechnique.multi_head]

    #NoFederatedLearning
    nets_types_list_NoFederatedLearning  = [NetsType.C_alex_S_alex]#,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]


    # parameters for PseudoLabelsClusters
    nets_types_list_PseudoLabelsClusters  = [NetsType.C_alex_S_vgg,NetsType.C_alex_S_alex]#,NetsType.C_alex_S_vgg]#,NetsType.C_alex_S_vgg]
    net_cluster_technique_list = [NetClusterTechnique.multi_model]#,NetClusterTechnique.multi_head]
    server_input_tech_list = [ServerInputTech.max]
    cluster_technique_list = [ClusterTechnique.greedy_elimination_L2]#[ClusterTechnique.greedy_elimination_cross_entropy]#[ClusterTechnique.manual_single_iter,ClusterTechnique.manual,ClusterTechnique.kmeans]
    server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster]#[ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]
    num_cluster_list = [1, "Optimal"]

    # parameters for fedAvg
    num_cluster_list_fedAVG = [1,"Optimal"] # dont touch
    nets_types_list_fedAVG  = [NetsType.C_alex_S_alex] # dont touch
    cluster_technique_list_fedAVG = [ClusterTechnique.kmeans] # we need this because of logic in num_cluster_list_fedAVG



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

                for server_split_ratio in server_split_ratio_list:
                    experiment_config.server_split_ratio = server_split_ratio
                    data_to_pickle[data_set.name][num_clients][num_opt_clusters][
                        server_split_ratio] = {}
                    experiment_config.mix_percentage = mix_percentage

                    for alpha_dicht in alpha_dichts:
                        experiment_config.alpha_dich =alpha_dicht
                        data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht] = {}
                        clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data = create_data()


                        for algorithm_selection in algorithm_selection_list:
                            experiment_config.algorithm_selection = algorithm_selection
                            if algorithm_selection == AlgorithmSelected.PseudoLabelsNoServerModel:
                                nets_types_list_PseudoLabelsClusters = [NetsType.C_alex]
                                net_cluster_technique_list = [NetClusterTechnique.no_model]
                                server_input_tech_list = [ServerInputTech.mean]
                                cluster_technique_list = [ClusterTechnique.kmeans]
                                server_feedback_technique_list = [ServerFeedbackTechnique.similar_to_cluster,
                                                                  ServerFeedbackTechnique.similar_to_client]  # [ServerFeedbackTechnique.similar_to_cluster,ServerFeedbackTechnique.similar_to_client]



                            data_to_pickle[data_set.name][num_clients][num_opt_clusters][server_split_ratio][
                            alpha_dicht][algorithm_selection.name] = {}
                            run_exp_by_algo()