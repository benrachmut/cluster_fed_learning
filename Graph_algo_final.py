from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected

def analize_PseudoLabelsClusters(algo):
    ans = {}
    for net_type in merged_dict_dich_algo.keys():
        algo_name = algo_names[algo]+","
        if net_type == NetsType.C_alex_S_vgg.name:
            dict_ = merged_dict_dich_algo[NetsType.C_alex_S_vgg.name]
            algo_name = algo_name+ "VGG,"
        if net_type == NetsType.C_alex_S_alex.name:
            dict_ = merged_dict_dich_algo[NetsType.C_alex_S_alex.name]
            algo_name =  algo_name+"AlexNet,"
        dict1 = dict_["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"]

        try:
            for weights_ in [WeightForPS.withWeights.name]:
                dict2 =dict1[weights_]
                for const in [InputConsistency.withInputConsistency.name]:
                    dict3=dict2[const]
                    for epsilon in dict3.keys():
                        rd = dict3[epsilon]
                        if top_what == 1:
                            ans[algo_name+ "Server" ] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)
                        if top_what == 5:
                            ans[algo_name+ "Server" ] = get_avg_of_entity(rd.server_accuracy_per_client_5_max)
                        if top_what == 10:
                            ans[algo_name+ "Server" ] = get_avg_of_entity(rd.server_accuracy_per_client_10_max)
                        #ans[algo_name+ "Clients"] = get_avg_of_entity(rd.client_accuracy_per_client_1)

        except:



                rd = dict1[0][WeightForPS.withWeights.name][InputConsistency.withInputConsistency.name]
                if top_what == 1:
                    ans[algo_name + "Server"] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)
                if top_what == 5:
                    ans[algo_name + "Server"] = get_avg_of_entity(rd.server_accuracy_per_client_5_max)
                if top_what == 10:
                    ans[algo_name + "Server"] = get_avg_of_entity(rd.server_accuracy_per_client_10_max)
                # ans[algo_name+ "Clients"] = get_avg_of_entity(rd.client_accuracy_per_client_1)

    return ans


def analize_PseudoLabelsNoServerModel(algo):
    ans = {}
    algo_name = algo_names[algo]
    dict_ = merged_dict_dich_algo["C_alex"]["no_model"]["mean"]["kmeans"]["similar_to_client"]
    for cluster in dict_.keys():
        if cluster == 1:
            algo_name = algo_name #+ cluster_names[cluster]
            rd = dict_[cluster]

            if top_what == 1:
                ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
            if top_what == 5:
                ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_5)
            if top_what == 10:
                ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_10)


    return ans

def analize_NoFederatedLearning(algo):
    ans = {}
    algo_name = algo_names[algo]

    rd = merged_dict_dich_algo["C_alex_S_alex"]
    fixed_dict = {}

    if top_what == 1:
        data_to_fix = rd.client_accuracy_per_client_1
    if top_what == 5:
        data_to_fix = rd.client_accuracy_per_client_5
    if top_what == 10:
        data_to_fix = rd.client_accuracy_per_client_10

    for client_id, dict_x_y in data_to_fix.items():
        dict_x_y_fixed = {}
        for x,y in dict_x_y.items():
            dict_x_y_fixed[x/5-1]=y
        fixed_dict[client_id]=dict_x_y_fixed
    ans[algo_name] = get_avg_of_entity(fixed_dict)
    return ans

def analize_FedAvg(algo):
    ans = {}
    algo_name = algo_names[algo]

    try:
        rd = merged_dict_dich_algo["C_alex_S_alex"]["multi_model"]["max"]["kmeans"]["similar_to_cluster"][1]
    except:
        rd = merged_dict_dich_algo["C_alex_S_alex"]["no_model"]["mean"]["kmeans"]["similar_to_cluster"][1]

    if top_what == 1:
        ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
    if top_what == 5:
        ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_5)
    if top_what == 10:
        ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_10)
    return ans

def analize_pFedCK(algo):
    ans = {}
    algo_name = algo_names[algo]
    for net_type in merged_dict_dich_algo.keys():
        rd = merged_dict_dich_algo[net_type]
        if top_what == 1:
            ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
        if top_what == 5:
            ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_5)
        if top_what == 10:
            ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_10)
        return ans


def get_data_per_algo(algo):

    if algo == AlgorithmSelected.PseudoLabelsClusters.name:
        return analize_PseudoLabelsClusters(algo)
    if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
        return  analize_PseudoLabelsNoServerModel(algo)
    if algo == AlgorithmSelected.NoFederatedLearning.name:
        return analize_NoFederatedLearning(algo)
    if algo == AlgorithmSelected.FedAvg.name:
        return analize_FedAvg(algo)
    if algo == AlgorithmSelected.pFedCK.name:
        return analize_pFedCK(algo)


def update_data(data, data_type):
    for algo, xy_dict in data.items():
        new_xy = {x + 1: y for x, y in xy_dict.items()}  # shift x keys by +1
        new_xy[0.0] = start_point[data_type]             # add new point at x = 0
        data[algo] = dict(sorted(new_xy.items()))        # optional: sort by x if desired

if __name__ == '__main__':
    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group
    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"CPL-Fed"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"FedMd",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                AlgorithmSelected.pFedCK.name:"pFedCK"
                }
    colors = {"CPL-Fed,VGG,Server": "blue",
              "CPL-Fed,AlexNet,Server": "red",
              "FedMd": "Green",
              "No FL": "Gray",
              "FedAvg": "brown",
              "pFedCK": "purple"}


    all_data = read_all_pkls("graph_algo_final")
    merged_dict1 = merge_dicts(all_data)
    top_what_list = [1,5,10]
    for top_what in top_what_list:
        for data_type in [DataSet.EMNIST_balanced.name]:
            merged_dict = merged_dict1[data_type][25][5][0.2][100]
            data_for_graph = {}
            merged_dict_dich = copy.deepcopy(merged_dict)
            for algo in merged_dict_dich.keys():
                merged_dict_dich_algo = merged_dict_dich[algo]
                feedback = get_data_per_algo(algo)
                data_for_graph.update(feedback)



            start_point = {DataSet.CIFAR100.name:1,DataSet.CIFAR10.name:10,DataSet.TinyImageNet.name:0.5,DataSet.EMNIST_balanced.name:2.13,DataSet.SVHN.name:10}
            update_data(data_for_graph,data_type)

            #for dich in [100]:
            #    t = {}
            #    for k in colors.keys():
            #        t[k] = data_for_graph[dich][k]
            if top_what == 1:
                y_label = "Top-1 Accuracy (%)"
            if top_what == 5:
                y_label = "Top-5 Accuracy (%)"
            if top_what == 10:
                y_label = "Top-10 Accuracy (%)"

            y_lim = None
            if data_type == DataSet.CIFAR10.name and top_what == 1:
                y_lim = [60,83]
            if data_type == DataSet.CIFAR10.name and top_what == 5:
                y_lim = [85,100]
            if data_type == DataSet.EMNIST_balanced.name and top_what == 1:
                y_lim = [80,95]

            create_algo_graph(data_for_graph, "Iteration", y_label, "figures","Algo_Comp"+data_type+"_top="+str(top_what),colors,y_lim)

