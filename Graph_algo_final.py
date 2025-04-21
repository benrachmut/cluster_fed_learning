from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected

def analize_PseudoLabelsClusters(dich):
    ans = {}
    for net_type in merged_dict_dich_algo.keys():
        algo_name = "CPL-Fed,"
        if net_type == NetsType.C_alex_S_vgg.name:
            dict_ = merged_dict_dich_algo[NetsType.C_alex_S_vgg.name]
            algo_name = algo_name+ "VGG,"
        if net_type == NetsType.C_alex_S_alex.name:
            dict_ = merged_dict_dich_algo[NetsType.C_alex_S_alex.name]
            algo_name =  algo_name+"AlexNet,"
        dict_ = dict_["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"]
        for epsilon in dict_.keys():
            #algo_name = algo_name + ",Clusters:"+str(5+epsilon)
            rd=dict_[epsilon]

            ans[algo_name+ "Server" ] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)
            ans[algo_name+ "Clients"] = get_avg_of_entity(rd.client_accuracy_per_client_1)



    return ans


def analize_PseudoLabelsNoServerModel(dich):
    ans = {}
    algo_name = "FedMd,"
    dict_ = merged_dict_dich_algo["C_alex"]["no_model"]["mean"]["kmeans"]["similar_to_client"]
    for cluster in dict_.keys():
        algo_name = algo_name + cluster_names[cluster]
        rd = dict_[cluster]
        ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
        algo_name = "FedMd,"

    return ans

def get_data_per_algo(algo,dich):

    if algo == AlgorithmSelected.PseudoLabelsClusters.name:
        return analize_PseudoLabelsClusters(dich)
    if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
        return  analize_PseudoLabelsNoServerModel(dich)

if __name__ == '__main__':
    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group


    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"CPL-Fed"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"FedMd",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                }


    all_data = read_all_pkls("graph_algo_final")
    merged_dict = merge_dicts(all_data)

    merged_dict = merged_dict["CIFAR100"][25][5][0.2]
    #ans = {}
    data_for_graph = {}#{"Server":[],"Clients":[]}
    for dich in [100]:
        data_for_graph[dich] = {}
        merged_dict_dich = merged_dict[dich]
        for algo in merged_dict_dich.keys():

            merged_dict_dich_algo = merged_dict_dich[algo]
            feedback = get_data_per_algo(algo,dich)#get_data_per_algo(algo,dich)


            data_for_graph[dich].update(feedback)
                #data_for_graph[dich]["Server"]=feedback["Server"]
                #data_for_graph[dich]["Clients"]=feedback["Clients"]
    print()

    for dich in [100]:
        create_CPL_graph(data_for_graph[dich], "Iteration", "Accuracy (%)", "figures","Iterations_CPL_"+str(dich))

