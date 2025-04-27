from main_ import *
from Graph_global import *
from config import AlgorithmSelected

def analize_PseudoLabelsClusters(dict_):
    #ans = {"Clients":{},"Server":{}}
    ans ={}
    for dist_from_opt_clusters in dict_.keys():
        #algo_name = algo_name + ",Clusters:"+str(5+epsilon)
        rd=dict_[dist_from_opt_clusters][WeightForPS.withWeights.name][InputConsistency.withInputConsistency.name]
        temp_ = get_avg_of_entity(rd.server_accuracy_per_client_1_max)
        ans [5+dist_from_opt_clusters] = max(temp_.values())
        #ans["Server"][5+dist_from_opt_clusters] = max(temp_.values())
        #temp_ = get_avg_of_entity(rd.client_accuracy_per_client_1)
        #ans["Clients"][5+dist_from_opt_clusters] = max(temp_.values())




    return ans



if __name__ == '__main__':

    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"CPL"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"CPL-NSM",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                }


    all_data = read_all_pkls("graph_clusters_final")
    merged_dict = merge_dicts(all_data)

    merged_dict = merged_dict["CIFAR100"][25][5][0.2]
    #ans = {}
    data_for_graph = {}#{"Server":[],"Clients":[]}
    for dich in [100]:
        data_for_graph[dich] = {}
        merged_dict_dich = merged_dict[dich]
        for algo in merged_dict_dich.keys():
            if algo == AlgorithmSelected.PseudoLabelsClusters.name:
                merged_dict_dich_algo = merged_dict_dich[algo][NetsType.C_alex_S_alex.name]["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"]
                if WeightForPS.withWeights.name in merged_dict_dich_algo:
                    del merged_dict_dich_algo[WeightForPS.withWeights.name]
                feedback = analize_PseudoLabelsClusters(merged_dict_dich_algo)#get_data_per_algo(algo,dich)
                data_for_graph[dich].update(feedback)
        print()

    for dich in [100]:
        data_for_graph[dich] = dict(sorted(data_for_graph[dich].items()))

        create_algo_cluster(data_for_graph[dich], "Clusters", "Max Accuracy (%)", "figures","Cluster_CPL_"+str(dich))

