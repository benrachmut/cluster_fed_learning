from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected


def get_data_per_iter(rd):
    ans = {}
    for client_id, data_per_iter in rd.server_accuracy_per_client_1_max.items():
        for iter_, number_ in data_per_iter.items():
            if iter_ not in ans:
                ans[iter_] = []
            ans[iter_].append(number_)
    return ans
def get_data_for_graph_cluster():
    ans = {}
    for cluster, rd_list in rds.items():
        for rd in rd_list:
            data_per_iter = get_data_per_iter(rd)
            max_iter = max(data_per_iter, key=lambda k: sum(data_per_iter[k]) / len(data_per_iter[k]))
            if cluster not in ans:
                ans[cluster] = []
            ans[cluster].extend(data_per_iter[max_iter])
    return ans

if __name__ == '__main__':

    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group

    all_data = read_all_pkls("data_cluster")#("data_algo")
    merged_dict1 = merge_dicts(all_data)
    #top_what_list = [1,5,10]

    #what_top_dict = {DataSet.CIFAR100.name:1,DataSet.CIFAR10.name:1,DataSet.TinyImageNet.name:5,DataSet.EMNIST_balanced.name:1}
    #for top_what in top_what_list:

    merged_dict = merged_dict1[DataSet.CIFAR100.name][25][5][0.2][5]
    rds = switch_algo_and_seed_cluster(merged_dict, dich=5,data_type=DataSet.CIFAR100.name)
    data_for_graph = get_data_for_graph_cluster()

    create_single_avg_plot(data_for_graph, x_label="Amount of Clusters", y_label="Top-1 Accuracy (%)")




        #data_for_graph[data_type] = collect_data_per_iteration(merged_dict,what_top_dict[data_type],data_type)

