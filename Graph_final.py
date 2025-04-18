from sympy.abc import epsilon

from Graph_global import *
from config import AlgorithmSelected

def analize_PseudoLabelsClusters():
    ans = {}
    for net_type in merged_dict_dich_algo.keys():
        algo_name = algo_names[AlgorithmSelected.PseudoLabelsClusters.name]
        if net_type == NetsType.C_alex_S_vgg.name:
            dict_ = merged_dict_dich_algo[NetsType.C_alex_S_vgg.name]
            algo_name = algo_name + "_S_vgg"
        if net_type == NetsType.C_alex_S_alex.name:
            dict_ = merged_dict_dich_algo[NetsType.C_alex_S_alex.name]
            algo_name = algo_name + "_S_alex"
        dict_ = dict_["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"]
        for epsilon in dict_.keys():
            algo_name = algo_name + "ε_"+str(epsilon)
            rd=dict_[epsilon]
            ans[algo_name] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)
    return ans


def get_data_per_algo(algo):

    if algo == AlgorithmSelected.PseudoLabelsClusters.name:
        return analize_PseudoLabelsClusters()

if __name__ == '__main__':

    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"CPL"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"CPL-NSM",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                }


    all_data = read_all_pkls("graph_final")
    merged_dict = merge_dicts(all_data)

    merged_dict = merged_dict["CIFAR100"][25][5][0.2]
    ans = {}

    for dich in [100]:
        ans[dich] = {}
        merged_dict_dich = merged_dict[dich]
        for algo in merged_dict_dich.keys():
            merged_dict_dich_algo = merged_dict_dich[algo]
            ans[dich] = get_data_per_algo(algo)

    data_for_graph = {}
