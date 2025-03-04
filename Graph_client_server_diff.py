from Graph_global import *
from config import *



def get_data_for_graph_algo_PseudoLabelsClusters(algo):

    for net_type in dict_per_algo.keys():
        all_clusters = dict_per_algo[net_type]["multi_model"]["max"]["kmeans"]["similar_to_cluster"]
        for cluster,rd in all_clusters.items():
            data_for_graph["Clients"][str(cluster)] = get_avg_of_entity(rd.client_accuracy_per_client_1)
            data_for_graph["Server Models"][str(cluster)] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)




def get_data_for_graph_algo_NoFederatedLearning(algo):

    for net_type in dict_per_algo.keys():
        algo_name = algo_names[algo]
        rd = dict_per_algo[net_type]
        data_for_graph["Clients"][algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)


if __name__ == '__main__':


    all_data = read_all_pkls("Graph_client_server_diff")
    merged_dict = merge_dicts(all_data)

    merged_dict = merged_dict["CIFAR100"][25][5][0.2][0.2]
    data_for_graph = {"Clients":{},"Server Models":{}}

    for algo in merged_dict.keys():
        dict_per_algo = merged_dict[algo]
        get_data_for_graph_algo_PseudoLabelsClusters(algo)




