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
        dict_ = dict_["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"]
        for epsilon in dict_.keys():
            #algo_name = algo_name + ",Clusters:"+str(5+epsilon)
            rd=dict_[epsilon]

            ans[algo_name+ "Server" ] = get_avg_of_entity(rd.server_accuracy_per_client_1_max)
            ans[algo_name+ "Clients"] = get_avg_of_entity(rd.client_accuracy_per_client_1)



    return ans


def analize_PseudoLabelsNoServerModel(algo):
    ans = {}
    algo_name = algo_names[algo]
    dict_ = merged_dict_dich_algo["C_alex"]["no_model"]["mean"]["kmeans"]["similar_to_client"]
    for cluster in dict_.keys():
        if cluster == 1:
            algo_name = algo_name #+ cluster_names[cluster]
            rd = dict_[cluster]
            ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)

    return ans

def analize_NoFederatedLearning(algo):
    ans = {}
    algo_name = algo_names[algo]

    rd = merged_dict_dich_algo["C_alex_S_alex"]
    fixed_dict = {}
    data_to_fix = rd.client_accuracy_per_client_1
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

    rd = merged_dict_dich_algo["C_alex_S_alex"]["multi_model"]["max"]["kmeans"]["similar_to_cluster"][1]
    ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
    return ans

def analize_pFedCK(algo):
    ans = {}
    algo_name = algo_names[algo]

    rd = merged_dict_dich_algo["C_alex_S_vgg"]
    ans[algo_name] = get_avg_of_entity(rd.client_accuracy_per_client_1)
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

if __name__ == '__main__':
    cluster_names = {"Optimal":"CBG",1:"No Clusters"} #Cluster By Group


    algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"CPL-Fed"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"FedMd",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                AlgorithmSelected.pFedCK.name:"pFedCK"
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
            feedback = get_data_per_algo(algo)#get_data_per_algo(algo,dich)


            data_for_graph[dich].update(feedback)
                #data_for_graph[dich]["Server"]=feedback["Server"]
                #data_for_graph[dich]["Clients"]=feedback["Clients"]
    print()

    colors = {"CPL-Fed,VGG,Server":"blue",
              "CPL-Fed,AlexNet,Server":"red",
              "FedMd":"Green",
              "No FL":"Gray",
            "FedAvg":"brown",
              "pFedCK":"purple"}



    for dich in [100]:
        t = {}
        for k in colors.keys():
            t[k] = data_for_graph[dich][k]
        create_algo_graph(t, "Iteration", "Accuracy (%)", "figures","Algo_Comp"+str(dich),colors)

