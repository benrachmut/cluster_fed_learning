import matplotlib.pyplot as plt

from Graph_global import *
from config import *



def get_data_for_graph_algo_PseudoLabelsClusters(algo):

    for net_type in dict_per_algo.keys():


        all_clusters = dict_per_algo[net_type]["multi_model"]["max"]["kmeans"]["similar_to_cluster"]
        for cluster,rd in all_clusters.items():
            if cluster == 5:
                if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
                    algo_name = algo_names[algo]+ ", "+"(Clients)"
                    data_for_graph[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
                if algo == AlgorithmSelected.PseudoLabelsClusters.name:
                    algo_name = algo_names[algo] + ", " + net_name[net_type]+", "+"(Server Models)"
                    data_for_graph[algo_name] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)

def get_data_for_graph_algo_PseudoLabelsNoServerModel(algo):
    all_clusters = dict_per_algo["C_alex"]["no_model"]["mean"]["kmeans"]["similar_to_cluster"]
    for cluster, rd in all_clusters.items():
        if cluster == 5 or cluster == 1:
            if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
                algo_name = algo_names[algo]+" Cluster: "+str(cluster)+" (Clients)"
                data_for_graph[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)

def get_data_for_graph_algo_NoFederatedLearning(algo):

    for net_type in dict_per_algo.keys():
        algo_name = algo_names[algo] +" (Clients)"

        rd = dict_per_algo[net_type]
        data_ = get_avg_of_entity(rd.client_accuracy_per_client_1)
        updated_data = {}
        for k,v in data_.items():
            updated_data[k/10 - 1] = v
        data_for_graph[algo_name] =updated_data


def get_data_for_graph_algo_Centralized(algo):
   dict_ = dict_per_algo["S_vgg"][NetClusterTechnique.multi_model]
   for cluster in ["Optimal"]:
       rd = dict_[cluster]
       algo_name = algo_names[algo] + ", " + "(Clusters:"+str(cluster)+")"
       data_ = get_avg_of_entity(rd.server_accuracy_per_cluster)
       updated_data = {}
       for k, v in data_.items():
           updated_data[k / 10 - 1] = v
       data_for_graph[algo_name] = updated_data

    #for net_type in dict_per_algo.keys():
    #    all_clusters = dict_per_algo[net_type]["multi_model"]["max"]["kmeans"]["similar_to_cluster"]
    #    for cluster, rd in all_clusters.items():
    #        if cluster == 5:
    #            if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
    #                algo_name = algo_names[algo] + ", " + "(Clients)"
    #                data_for_graph[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
    #            if algo == AlgorithmSelected.PseudoLabelsClusters.name:
    #                algo_name = algo_names[algo] + ", " + net_name[net_type] + ", " + "(Server Models)"
    #                data_for_graph[algo_name] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)


def get_data_for_graph_algo_FedAvg(algo):
    algo_name = algo_names[algo] + " (Clients)"
    rd = dict_per_algo["C_alex_S_alex"]["multi_model"]["max"]["kmeans"]["similar_to_cluster"]["Optimal"]

    data_ = get_avg_of_entity(rd.client_accuracy_per_client_1)
    updated_data = {}
    for k, v in data_.items():
        updated_data[k] = v
    data_for_graph[algo_name] = updated_data


if __name__ == '__main__':

    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"C-PL"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"C-PL-NSM",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                }


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
        if algo == AlgorithmSelected.Centralized.name:
            get_data_for_graph_algo_Centralized(algo)
        if algo == AlgorithmSelected.FedAvg.name:
            get_data_for_graph_algo_FedAvg(algo)

    plt.figure(figsize=(7, 5))

    # Define a color cycle for different algorithms
    colors = ["red", "orange", "green","blue" , "purple","Gray","brown"]
    for i, (algorithm, points) in enumerate(data_for_graph.items()):
        x_values = list(points.keys())
        y_values = list(points.values())
        plt.plot(x_values, y_values, label=algorithm, color=colors[i % len(colors)])

    plt.xlabel("Iteration")
    plt.ylabel("Average Accuracy")
    plt.title("Algorithm Performance")

    # Move the legend outside the plot
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjust layout to make room for the legend

    plt.xlim(0,9)
    plt.savefig("algorithm_comparison.png", bbox_inches="tight", dpi=300)

    #plt.show()