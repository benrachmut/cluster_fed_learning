from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected




















if __name__ == '__main__':

    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group

    all_data = read_all_pkls("tttt")#("data_algo")
    merged_dict1 = merge_dicts(all_data)
    #top_what_list = [1,5,10]

    what_top_dict = {DataSet.CIFAR100.name:1,DataSet.CIFAR10.name:1,DataSet.TinyImageNet.name:5,DataSet.EMNIST_balanced.name:1}
    #for top_what in top_what_list:
    data_for_graph={}
    for dich  in [5]:
        for data_type in [DataSet.CIFAR100.name]:#[DataSet.CIFAR100.name,DataSet.CIFAR10.name,DataSet.TinyImageNet.name,DataSet.EMNIST_balanced.name]:
            merged_dict = merged_dict1[data_type][25][5][0.2][dich]
            merged_dict = switch_algo_and_seed(merged_dict, dich=dich,data_type=data_type)
            data_for_graph[data_type] = collect_data_per_iteration(merged_dict,what_top_dict[data_type],data_type)


        y_label_dict = {DataSet.CIFAR100.name:"Top-1 Accuracy (%)",DataSet.CIFAR10.name:"Top-1 Accuracy (%)",DataSet.TinyImageNet.name:"Top-5 Accuracy (%)",DataSet.EMNIST_balanced.name:"Top-1 Accuracy (%)"}
        y_lim = {DataSet.CIFAR100.name:[0,37],DataSet.CIFAR10.name:[60,80],DataSet.TinyImageNet.name:[0,40],DataSet.EMNIST_balanced.name:[60,95]}
        #y_lim = None
        #if data_type == DataSet.CIFAR10.name and top_what == 1:
        #    y_lim = [60,83]
        #if data_type == DataSet.CIFAR10.name and top_what == 5:
        #    y_lim = [85,100]
        #if data_type == DataSet.EMNIST_balanced.name and top_what == 1:
        #    y_lim = [80,95]
        create_2x2_algo_grid(data_for_graph, x_label ="Iteration" , y_label_dict=y_label_dict, y_lim_dict=y_lim, confidence=0.95, dich = dich)

        #create_algo_graph(data_for_graph, "Iteration", y_label, "figures","Algo_Comp"+data_type+"_top="+str(top_what),colors,y_lim)

