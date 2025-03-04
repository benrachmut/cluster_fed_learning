from Graph_global import *
from config import *



def get_data_for_graph_algo_PseudoLabelsClusters(algo):

    for net_type in dict_per_algo.keys():


        all_clusters = dict_per_algo[net_type]["multi_model"]["max"]["kmeans"]["similar_to_cluster"]
        for cluster,rd in all_clusters.items():
            if cluster == 5:
                if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
                    algo_name = algo_names[algo]
                    data_for_graph[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
                if algo == AlgorithmSelected.PseudoLabelsClusters.name:
                    algo_name = algo_names[algo] + ", " + net_name[net_type]
                    data_for_graph[algo_name] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)

def get_data_for_graph_algo_PseudoLabelsNoServerModel(algo):
    all_clusters = dict_per_algo["C_alex"]["no_model"]["mean"]["kmeans"]["similar_to_cluster"]
    for cluster, rd in all_clusters.items():
        if cluster == 5 or cluster == 1:
            if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
                algo_name = algo_names[algo]+" Cluster: "+str(cluster)
                data_for_graph[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)

def get_data_for_graph_algo_NoFederatedLearning(algo):

    for net_type in dict_per_algo.keys():
        algo_name = algo_names[algo]

        rd = dict_per_algo[net_type]
        data_for_graph[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)


if __name__ == '__main__':

    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"C-PL",AlgorithmSelected.PseudoLabelsNoServerModel.name:"C-PL-NSM",AlgorithmSelected.NoFederatedLearning.name:"No FL"  }


    all_data = read_all_pkls("Graph_diff_algos")
    merged_dict = merge_dicts(all_data)

    merged_dict = merged_dict["CIFAR100"][25][5][0.2][0.2]
    data_for_graph = {}

    for algo in merged_dict.keys():
        dict_per_algo = merged_dict[algo]
        if algo == AlgorithmSelected.PseudoLabelsClusters.name :
            get_data_for_graph_algo_PseudoLabelsClusters(algo)

        if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
            get_data_for_graph_algo_PseudoLabelsNoServerModel(algo)
        if algo == AlgorithmSelected.NoFederatedLearning.name:
            get_data_for_graph_algo_NoFederatedLearning(algo)

    print()
