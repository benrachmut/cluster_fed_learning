from sympy.abc import epsilon
from torch.ao.quantization.backend_config.native import weight_only_quint8_dtype_config

from main_ import *
from Graph_global import *
from config import AlgorithmSelected

def analize_PseudoLabelsClusters(dich):
    ans = {}
    for net_type in merged_dict_dich_algo.keys():
        algo_name = ""#"α:"+str(dich)#algo_names[AlgorithmSelected.PseudoLabelsClusters.name]
        if net_type == NetsType.C_alex_S_vgg.name:
            dict_ = merged_dict_dich_algo[NetsType.C_alex_S_vgg.name]
            algo_name =  "VGG"
        if net_type == NetsType.C_alex_S_alex.name:
            dict_ = merged_dict_dich_algo[NetsType.C_alex_S_alex.name]
            algo_name =  "AlexNet"
        dict_ = dict_["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"]
        for weights_ in list(WeightForPS):

            dict_1 = dict_[weights_.name]
            for consistency in list(InputConsistency):
                rd = dict_1[consistency.name][0]
                algo_name_a = algo_name+"_"+weights_.name+"_"+consistency.name
                    #algo_name = algo_name + ",Clusters:"+str(5+epsilon)

                if "Server"  not in ans:
                    ans["Server"] = {}
                if "Clients" not in ans:
                    ans["Clients"] = {}
                ans["Server" ][algo_name_a] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)
                ans["Clients"] [algo_name_a] = get_avg_of_entity(rd.client_accuracy_per_client_1)



    return ans


def get_data_per_algo(algo,dich):

    if algo == AlgorithmSelected.PseudoLabelsClusters.name:
        return analize_PseudoLabelsClusters(dich)

if __name__ == '__main__':

    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"CPL"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"CPL-NSM",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                }


    all_data = read_all_pkls("graph_variants_final")
    merged_dict = merge_dicts(all_data)

    merged_dict = merged_dict["CIFAR100"][25][5][0.2]
    #ans = {}
    data_for_graph = {}#{"Server":[],"Clients":[]}
    for dich in [100]:
        data_for_graph[dich] = {}
        merged_dict_dich = merged_dict[dich]
        for algo in merged_dict_dich.keys():
            if algo == AlgorithmSelected.PseudoLabelsClusters.name:

                merged_dict_dich_algo = merged_dict_dich[algo]
                feedback = analize_PseudoLabelsClusters(dich)#get_data_per_algo(algo,dich)
                data_for_graph[dich]["Server"]=feedback["Server"]
                data_for_graph[dich]["Clients"]=feedback["Clients"]


    for dich in [100]:
        create_CPL_graph(data_for_graph[dich], "Iteration", "Accuracy (%)", "figures","Iterations_CPL_"+str(dich))

