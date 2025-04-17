from Graph_global import *
from config import AlgorithmSelected

if __name__ == '__main__':

    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"C-PL"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"C-PL-NSM",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                }


    all_data = read_all_pkls("graph_final")
    merged_dict = merge_dicts(all_data)

    merged_dict = merged_dict["CIFAR100"][25][5][0.2]
    for dich in [0.5,1,5,10,100]
    data_for_graph = {}
