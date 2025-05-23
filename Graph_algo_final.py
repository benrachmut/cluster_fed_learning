from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected




















if __name__ == '__main__':

    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group

    all_data = read_all_pkls("data_algo/100")#("data_algo")
    merged_dict1 = merge_dicts(all_data)
    #top_what_list = [1,5,10]

    what_top_dict = {DataSet.CIFAR100.name:1,DataSet.CIFAR10.name:1,DataSet.TinyImageNet.name:5,DataSet.EMNIST_balanced.name:1}
    #for top_what in top_what_list:
    data_for_graph={}
    for dich  in [100]:
        for data_type in [DataSet.CIFAR100.name,DataSet.CIFAR10.name,DataSet.TinyImageNet.name,DataSet.EMNIST_balanced.name,]:
            merged_dict = merged_dict1[data_type][25][5][0.2][dich]
            merged_dict = switch_algo_and_seed(merged_dict, dich=dich,data_type=data_type)
            data_for_graph[data_type] = collect_data_per_iteration(merged_dict,what_top_dict[data_type],data_type)


        y_label_dict = {DataSet.CIFAR100.name:"Top-1 Accuracy (%)",DataSet.CIFAR10.name:"Top-1 Accuracy (%)",DataSet.TinyImageNet.name:"Top-5 Accuracy (%)",DataSet.EMNIST_balanced.name:"Top-1 Accuracy (%)"}
        y_lim = {DataSet.CIFAR100.name:[0,37],DataSet.CIFAR10.name:[60,80],DataSet.TinyImageNet.name:[0,40],DataSet.EMNIST_balanced.name:[70,95]}


        ttt = {}
        for data_type, d1 in data_for_graph.items():
            ttt[data_type] ={}
            for algo, d2 in d1.items():
                ttt[data_type][algo] ={}
                for i,l in d2.items():
                    ttt[data_type][algo][i] = sum(l)/len(l)
        #create_2x2_algo_grid(data_for_graph, x_label ="Iteration" , y_label_dict=y_label_dict, y_lim_dict=y_lim, confidence=0.95, dich = dich)
        create_1x4_algo_grid(data_for_graph, x_label ="Iteration" , y_label_dict=y_label_dict, y_lim_dict=y_lim, confidence=0.95, dich = dich)
        #create_algo_graph(data_for_graph, "Iteration", y_label, "figures","Algo_Comp"+data_type+"_top="+str(top_what),colors,y_lim)

