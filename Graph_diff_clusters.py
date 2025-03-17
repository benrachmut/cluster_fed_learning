import itertools

from Graph_global import *
from config import *



def get_data_for_graph_algo_PseudoLabelsClusters(cluster_tech):

    rd = dict_per_cluster["similar_to_cluster"][5]

    data_for_graph["Clients"][cluster_tech] = get_avg_of_entity(rd.client_accuracy_per_client_1)
    data_for_graph["Server Models"][cluster_tech] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)




if __name__ == '__main__':


    all_data = read_all_pkls("Graph_diff_clusters")
    merged_dict = merge_dicts(all_data)

    merged_dict = merged_dict["CIFAR100"][25][5][0.2][0.2]["PseudoLabelsClusters"]["C_alex_S_vgg"]["multi_model"]["max"]
    data_for_graph = {"Clients":{},"Server Models":{}}

    for cluster_tech in merged_dict.keys():
        dict_per_cluster = merged_dict[cluster_tech]
        get_data_for_graph_algo_PseudoLabelsClusters(cluster_tech)

    cluster_colors = {"kmeans": "red", "manual": "blue", "manual_single_iter": "green"}  # Adjust as needed
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
    plt.ylabel("Average Accuracy of Server Models/Clients ")
    plt.title("C-PL cluster techniques for K = 5, Server Models = VGG16")

    # Move legend outside the graph
    plt.legend(title="Entity and clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("clusters_tech_legend.png", bbox_inches="tight", dpi=300)  # Save the plot as PNG
    plt.show()