from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected









if __name__ == '__main__':

    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group

    all_data = read_all_pkls("data_cluster")#("data_algo")
    merged_dict1 = merge_dicts(all_data)
    #top_what_list = [1,5,10]

    #what_top_dict = {DataSet.CIFAR100.name:1,DataSet.CIFAR10.name:1,DataSet.TinyImageNet.name:5,DataSet.EMNIST_balanced.name:1}
    #for top_what in top_what_list:
    data_for_graph={}
    for data_type in [DataSet.CIFAR100.name,DataSet.CIFAR10.name,DataSet.TinyImageNet.name,DataSet.EMNIST_balanced.name]:
        merged_dict = merged_dict1[data_type][25][5][0.2][5]
        merged_dict = switch_algo_and_seed_cluster(merged_dict, dich=5,data_type=data_type)
        print()
        #data_for_graph[data_type] = collect_data_per_iteration(merged_dict,what_top_dict[data_type],data_type)

