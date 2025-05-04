from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected



def extract_rd_PseudoLabelsClusters(algo,dict_):
    ans = {}

    names = get_PseudoLabelsClusters_name(algo, dict_)

    for name in names:
        if name == algo_names[algo]+",AlexNet":
            dict_1 = dict_[NetsType.C_alex_S_alex.name]
        if name == algo_names[algo]+",VGG":
            dict_1 = dict_[NetsType.C_alex_S_vgg.name]
        rd = dict_1["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"][0][
                WeightForPS.withWeights.name][InputConsistency.withInputConsistency.name]

        ans[name] = rd

    return ans


def extract_rd_PseudoLabelsNoServerModel(algo,dict_):

    return dict_["C_alex"]["no_model"]["mean"]["kmeans"]["similar_to_client"][1]








def extract_rd_FedAvg(algo,dict_):
    return dict_[NetsType.C_alex_S_alex.name]["multi_model"]["max"]["kmeans"]["similar_to_cluster"][1]

def extract_rd_pFedCK(algo,dict_):
    for net_type in dict_.keys():
        return dict_[net_type]




def extract_rd(algo,dict_):
    if algo == AlgorithmSelected.PseudoLabelsClusters.name:
        return extract_rd_PseudoLabelsClusters(algo,dict_)
    if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
        return extract_rd_PseudoLabelsNoServerModel(algo,dict_)
    if algo == AlgorithmSelected.NoFederatedLearning.name:
        return dict_["C_alex_S_alex"]
    if algo == AlgorithmSelected.FedAvg.name:
        return extract_rd_FedAvg(algo,dict_)
    if algo == AlgorithmSelected.pFedCK.name:
        return extract_rd_pFedCK(algo,dict_)


def get_PseudoLabelsClusters_name(algo,dict_):
    ans = []
    for net_type in dict_.keys():
        algo_name = algo_names[algo] + ","
        if net_type == NetsType.C_alex_S_vgg.name:
            algo_name = algo_name + "VGG"
        if net_type == NetsType.C_alex_S_alex.name:
            algo_name = algo_name + "AlexNet"
        ans.append(algo_name)
    return  ans

def switch_algo_and_seed(merged_dict,dich):
    rds = {}
    for seed in seeds_dict[dich][data_type]:
        for algo in merged_dict[seed]:
            algo_name = algo_names[algo]
            if algo == AlgorithmSelected.PseudoLabelsClusters.name:
                algo_name_list = get_PseudoLabelsClusters_name(algo,merged_dict[seed][algo])
                for name_ in algo_name_list:
                    if name_ not in rds.keys() :
                        rds[name_] = []

            elif algo_name not in rds.keys() :
                rds[algo_name] = []
            rd_output = extract_rd(algo, merged_dict[seed][algo])
            if isinstance(rd_output,dict):
                for k,v in rd_output.items():
                    if k not in rds:
                        rds[k]=[]
                    rds[k].append(v)
            else:
                rds[algo_name].append(rd_output)



            #ans[algo].append(merged_dict[seed][algo])
    return rds

def get_data_per_client_client(rd):
    if top_what == 1:
        return rd.client_accuracy_per_client_1
    if top_what == 5:
        return  rd.client_accuracy_per_client_5
    if top_what == 10:
        return  rd.client_accuracy_per_client_10
def fix_data_NoFederatedLearning(data_per_client):
    ans = {}
    for client_id, dict_x_y in data_per_client.items():
        dict_x_y_fixed = {}
        for x, y in dict_x_y.items():
            dict_x_y_fixed[x / 5 - 1] = y
        dict_x_y_fixed[max(dict_x_y_fixed.keys())+1] = dict_x_y_fixed[max(dict_x_y_fixed.keys())]
        ans[client_id] = dict_x_y_fixed
    return ans
def get_data_per_client_server(rd):
    if top_what == 1:
        return rd.server_accuracy_per_client_1_max
    if top_what == 5:
        return  rd.server_accuracy_per_client_5_max
    if top_what == 10:
        return  rd.server_accuracy_per_client_10_max
def get_data_per_client(rd,algo):
    if algo == algo_names[AlgorithmSelected.NoFederatedLearning.name]:
        data_per_client = get_data_per_client_client(rd)
        data_per_client = fix_data_NoFederatedLearning(data_per_client)
    if algo == algo_names[AlgorithmSelected.pFedCK.name] or algo == algo_names[
        AlgorithmSelected.PseudoLabelsNoServerModel.name]  or algo == algo_names[AlgorithmSelected.FedAvg.name]:
        data_per_client = get_data_per_client_client(rd)
    if algo ==algo_names[AlgorithmSelected.PseudoLabelsClusters.name]+",VGG" or algo ==algo_names[AlgorithmSelected.PseudoLabelsClusters.name]+",AlexNet":
        data_per_client = get_data_per_client_server(rd)
    update_data(data_per_client, data_type)
    return data_per_client
def collect_data_per_iteration(merged_dict):
    ans = {}
    for algo, rd_list in merged_dict.items():
        data_per_iteration = {}
        for rd in rd_list:
            data_per_client = get_data_per_client(rd,algo)
            for client_id, data_dict in data_per_client.items():
                for iter_, v in data_dict.items():
                    if iter_ not in data_per_iteration.keys():
                        data_per_iteration[iter_] = []
                    data_per_iteration[iter_].append(v)
        ans[algo] = data_per_iteration
    return ans

if __name__ == '__main__':

    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group




    all_data = read_all_pkls("data_algo")
    merged_dict1 = merge_dicts(all_data)
    top_what_list = [1,5,10]
    for top_what in top_what_list:
        for data_type in [DataSet.CIFAR100.name]:
            for dich in [5]:
                merged_dict = merged_dict1[data_type][25][5][0.2][dich]
                merged_dict = switch_algo_and_seed(merged_dict,dich)
                data_for_graph = collect_data_per_iteration(merged_dict)
                print()




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

