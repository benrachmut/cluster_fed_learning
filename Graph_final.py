from Graph_global import *
from config import AlgorithmSelected

def analize_PseudoLabelsClusters():
    for net_type in merged_dict_dich.keys():
        if net_type == NetsType.C_alex_S_vgg.name:

        if net_type == NetsType.C_alex_S_alex.name:


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
    for dich in [0.5,1,5,10,100]:
        merged_dict_dich = merged_dict[dich]
        ans = {}
        for algo in merged_dict_dich.keys():
            dict_return = get_data_per_algo(algo)

    data_for_graph = {}
