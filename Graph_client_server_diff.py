import itertools

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

    cluster_colors = {"1": "red", "5": "blue", "Optimal": "green"}  # Adjust as needed
    linestyles = {"Clients": "dashed", "Server Models": "solid"}  # Different line styles for clients and server

    plt.figure(figsize=(7, 5))

    # Iterate through the dictionary
    for category, clusters in data_for_graph.items():
        for cluster, points in clusters.items():
            x_values = list(points.keys())
            y_values = list(points.values())
            color = cluster_colors.get(cluster, "black")  # Default to black if cluster is missing from color dict
            plt.plot(x_values, y_values, label=f"{category} - {cluster}",
                     linestyle=linestyles[category], color=color)

    plt.xlabel("Iteration")
    plt.ylabel("Average Accuracy of Server Models")
    plt.title("C-PL, Server Models = VGG16")

    # Move legend outside the graph
    #plt.legend(title="Entity and clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    #plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("clients_server_clusters.png", bbox_inches="tight", dpi=300)  # Save the plot as PNG
    plt.show()