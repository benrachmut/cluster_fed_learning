from Graph_global import *
from main_ import *


def switch_algo_and_seed_client_server(merged_dict, dich):
    rds = {}
    for seed in seeds_dict[dich][data_type]:
        for algo in merged_dict[seed]:
            algo_name = algo_names[algo]

            algo_name_list = get_PseudoLabelsClusters_name(algo,merged_dict[seed][algo])
            for name_ in algo_name_list:
                if name_ not in rds.keys() :
                    name_to_place = ""
                    if name_ == "MAPL,VGG":
                        name_to_place = "VGG"
                    else:
                        name_to_place = "AlexNet"

                    rds[name_to_place] = []
                rd_output = extract_rd_PseudoLabelsClusters_server_client(algo,merged_dict[seed][algo])#extract_rd(algo, )
                for k,v in rd_output.items():
                    if k not in rds:
                        rds[k]=[]
                    rds[k].append(v)
    return rds


if __name__ == '__main__':

    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group

    all_data = read_all_pkls("diff_nets")
    merged_dict1 = merge_dicts(all_data)
    top_what_list = [1,5,10]
    data_type = DataSet.CIFAR100.name
    top_what = 1
    data_for_graph = {}
    for dich in [1]:
        merged_dict = merged_dict1[data_type][25][5][1][dich]
        merged_dict = switch_algo_and_seed(merged_dict,dich,data_type)
        new_name_dict = {}
        for k,v in merged_dict.items():
            new_name_dict[k]=v
        data_for_graph[dich]= collect_data_per_server_client_iteration(new_name_dict,top_what,data_type)
    print()




    the_plot = plot_model_server_client(data_for_graph)#(data_for_graph, "Iteration", y_label, "figures","Algo_Comp" + data_type + "_top=" + str(top_what))

