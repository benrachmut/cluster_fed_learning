# File path to your pickle file
import pickle
from main_ import *
from config import *

import numpy as np

#DIFF NETS, lr

##########--------------NON iid

# No server net:
#file_path ="weight_mem_True_batch_128_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_0.2_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netFalses_netVGGc_netALEXNET.pkl"
#server = alex
#file_path ="weight_mem_True_batch_128_trainLR_0.0001_tuneLR_0.001_Mix_Percent_0.2_Epoch_20_Server_Split_0.2_Num_Class_10_same_Clients_1with_server_netTrueserver_netALEXNETclient_netALEXNET.pkl"
#server = VGG
#file_path ="weight_mem_True_batch_128_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_0.2_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netVGGc_netALEXNET.pkl"


##########--------------iid

# No server net:
#file_path ="weight_mem_True_batch_64_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netFalses_netVGGc_netALEXNET.pkl"
#server = alex
#file_path ="weight_mem_True_batch_64_trainLR_0.0001_tuneLR_0.001_Mix_Percent_1_Epoch_20_Server_Split_0.2_Num_Class_10_same_Clients_1with_server_netTrueserver_netALEXNETclient_netALEXNET.pkl"
#server = VGG
#file_path ="weight_mem_True_batch_64_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netVGGc_netALEXNET.pkl"



#file_path ="04125/weight_mem_True_batch_64_trainLR_0.0001_tuneLR_0.001_Mix_Percent_1_Epoch_20_Server_Split_0.2_Num_Class_10_same_Clients_1with_server_netTrueserver_netVGGclient_netALEXNET.pkl"
#file_path ="04125/weight_mem_True_batch_128_trainLR_0.0001_tuneLR_0.001_Mix_Percent_0.2_Epoch_20_Server_Split_0.2_Num_Class_10_same_Clients_1with_server_netTrueserver_netVGGclient_netALEXNET.pkl"

#file_path ="04125/weight_mem_True_batch_64_trainLR_0.0001_tuneLR_0.001_Mix_Percent_1_Epoch_20_Server_Split_0.2_Num_Class_10_same_Clients_1with_server_netTrueserver_netALEXNETclient_netALEXNET.pkl"
#file_path ="04125/weight_mem_True_batch_128_trainLR_0.0001_tuneLR_0.001_Mix_Percent_0.2_Epoch_20_Server_Split_0.2_Num_Class_10_same_Clients_1with_server_netTrueserver_netALEXNETclient_netALEXNET.pkl"

#file_path ="04125/weight_mem_True_batch_64_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netVGGc_netALEXNET.pkl"
#file_path ="weight_mem_True_batch_128_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_0.2_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netVGGc_netALEXNET.pkl"

def get_all_iterations(data_dict,max_iter):

    all_iterations = set()
    for client_data in data_dict.values():
        all_iterations.update(client_data.keys())
    ans = []
    for iter in all_iterations:
        if iter<max_iter:
            ans.append(iter)
    return sorted(ans)

def plot_individual_clients(data_dict, all_accuracies,clients_to_remove,min_iter):
    """
    Plots the accuracy curves for individual clients and populates data for averaging.

    Args:
        data_dict (dict): Client accuracy data.
        all_accuracies (dict): Dictionary to store accuracies for averaging.
    """
    for client_id, accuracy_dict in data_dict.items():
        if isinstance(client_id,int):
            if client_id not in clients_to_remove:
                iterations = sorted(accuracy_dict.keys())
                accuracies = [accuracy_dict[iteration] for iteration in iterations]

                # Add accuracies to all_accuracies for averaging
                for iteration, acc in zip(iterations, accuracies):
                    if iteration<min_iter:
                        all_accuracies[iteration].append(acc)

        else:
            iterations = sorted(accuracy_dict.keys())
            iter_=[]
            for i in range(len(iterations)):
                if i<min_iter:
                    iter_.append(iterations[i])


            accuracies = [accuracy_dict[iteration] for iteration in iterations]
            acc_ = []
            for i in range(len(accuracies)):
                if i < min_iter:
                    acc_.append(accuracies[i])
            # Add accuracies to all_accuracies for averaging
            #for iteration, acc in zip(iterations, accuracies):
            #

            #        all_accuracies[iteration].append(acc)


            plt.plot(
                iter_, acc_, label=f"SERVER", marker='o', linestyle='-'
            )


def plot_individual_clients(data_dict, all_accuracies,clients_to_remove,min_iter):
    """
    Plots the accuracy curves for individual clients and populates data for averaging.

    Args:
        data_dict (dict): Client accuracy data.
        all_accuracies (dict): Dictionary to store accuracies for averaging.
    """
    for client_id, accuracy_dict in data_dict.items():
        if isinstance(client_id,int):
            if client_id not in clients_to_remove:
                iterations = sorted(accuracy_dict.keys())
                accuracies = [accuracy_dict[iteration] for iteration in iterations]

                # Add accuracies to all_accuracies for averaging
                for iteration, acc in zip(iterations, accuracies):
                    if iteration<min_iter:
                        all_accuracies[iteration].append(acc)

        else:
            iterations = sorted(accuracy_dict.keys())
            iter_=[]
            for i in range(len(iterations)):
                if i<min_iter:
                    iter_.append(iterations[i])


            accuracies = [accuracy_dict[iteration] for iteration in iterations]
            acc_ = []
            for i in range(len(accuracies)):
                if i < min_iter:
                    acc_.append(accuracies[i])
            # Add accuracies to all_accuracies for averaging
            #for iteration, acc in zip(iterations, accuracies):
            #

            #        all_accuracies[iteration].append(acc)


            plt.plot(
                iter_, acc_, label=f"SERVER", marker='o', linestyle='-'
            )

def plot_average_curve(all_accuracies, all_iterations,label_):
    """
    Plots the average accuracy curve based on data from all clients.

    Args:
        all_accuracies (dict): Dictionary of accuracies for each iteration.
        all_iterations (list): List of all unique iterations.
    """
    average_accuracies = [
        np.mean(all_accuracies[iteration]) for iteration in all_iterations
    ]
    plt.plot(
        all_iterations, average_accuracies, label=label_, color='black', linestyle='--', linewidth=2
    )

def finalize_plot(data_type_):
    """
    Adds labels, legend, grid, and displays the plot.
    """
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Validation ** "+data_type_+" **")
    #plt.legend(title="Entity")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def get_all_accuracies(data_dict, all_accuracies, clients_to_remove,all_iterations):

        for client_id, accuracy_dict in data_dict.items():
            if isinstance(client_id, int):
                if client_id not in clients_to_remove:
                    iterations = all_iterations
                    accuracies = [accuracy_dict[iteration] for iteration in iterations]
                    indexes_with_trouble = []
                    all_good_flag = False
                    for sing_acc in accuracies:
                        if sing_acc>12:
                            all_good_flag = True
                            break
                    if  all_good_flag:
                        for i in range(len(accuracies)):
                            if accuracies[i]<12 and i>0:
                                if i>0:
                                    accuracies[i] = accuracies[i-1]
                                else:
                                    for j in range(i+1,len(accuracies)):
                                        if accuracies[i]>12:
                                            accuracies[i] = accuracies[j]
                                            break






                                # Add accuracies to all_accuracies for averaging
                    for iteration, acc in zip(iterations, accuracies):
                        all_accuracies[iteration].append(acc)


def plot_client_accuracies(data_dict,cleints_to_remove,max_iter):
    """
    Plots the accuracy of clients over iterations and the average accuracy curve.

    Args:
        data_dict (dict): A dictionary where:
                          - Key: Client ID (e.g., int or str)
                          - Value: Dictionary with key as iteration (int) and value as accuracy (float)
    """
    all_iterations = get_all_iterations(data_dict)
    all_accuracies =  {iteration: [] for iteration in all_iterations}
    #get_all_accuracies()


    plot_individual_clients(data_dict, all_accuracies,cleints_to_remove,max_iter)
    plot_average_curve(all_accuracies, all_iterations)

    finalize_plot()

def get_server_data(data_dict,max_iterations):
    accuracy_list = list(data_dict["server"].values())
    ans = []
    for i in range(len(accuracy_list)):
        if i<max_iterations:
            ans.append(accuracy_list[i])
    return ans

class data_type_enum(Enum):
    non_iid = 1
    iid = 2

def get_name_file_dict_with_mem0401(folder_name, data_type):

    if data_type== data_type_enum.non_iid:
        return  {"No server":
             folder_name+"/weight_mem_True_batch_128_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_0.2_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netFalses_netVGGc_netALEXNET.pkl"
         ,
         "S.AlexNet":
             folder_name+"/weight_mem_True_batch_128_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_0.2_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netVGGc_netALEXNET.pkl"
         ,
         "S.VGG16" :
         folder_name+"/weight_mem_True_batch_128_trainLR_0.0001_tuneLR_0.001_Mix_Percent_0.2_Epoch_20_Server_Split_0.2_Num_Class_10_same_Clients_1with_server_netTrueserver_netVGGclient_netALEXNET.pkl"
         }

    if data_type== data_type_enum.iid:
        return {"No server":
             folder_name+"/weight_mem_True_batch_64_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netFalses_netVGGc_netALEXNET.pkl"
            ,
         "S.AlexNet":
               folder_name+"/fix/weight_mem_True_batch_64_trLRC_0.001_trainLR_0.001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netALEXNETc_netALEXNET.pkl"
                #folder_name + "/fix/weight_mem_True_batch_64_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netALEXNETc_netALEXNET.pkl"

               #folder_name+"/weight_mem_True_batch_64_trainLR_0.0001_tuneLR_0.001_Mix_Percent_1_Epoch_20_Server_Split_0.2_Num_Class_10_same_Clients_1with_server_netTrueserver_netALEXNETclient_netALEXNET.pkl"
            ,
         "S.VGG16":
             folder_name + "/weight_mem_True_batch_64_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netVGGc_netALEXNET.pkl"
         }


def get_name_file_dict_with_mem0501(folder_name, data_type):

    if data_type== data_type_enum.non_iid:
        return  {"No server":
             folder_name+"/weight_mem_True_batch_128_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_0.2_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netFalses_netVGGc_netALEXNET.pkl"
         ,
         "S.AlexNet":
             folder_name+"/weight_mem_True_batch_128_trLRC_0.001_trainLR_0.001_tuneLR_0.001_Mix_P_0.2_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netALEXNETc_netALEXNET.pkl"
         ,
         "S.VGG16" :
         folder_name+"/weight_mem_True_batch_128_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_0.2_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netVGGc_netALEXNET.pkl"
         }

    if data_type== data_type_enum.iid:
        return {"No server":
             folder_name+"/weight_mem_True_batch_64_trLRC_0.001_trainLR_0.001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netFalses_netALEXNETc_netALEXNET.pkl"
            ,
         "S.AlexNet":
               folder_name+"/weight_mem_True_batch_64_trLRC_0.001_trainLR_0.001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netALEXNETc_netALEXNET.pkl"

            ,
         "S.VGG16":
             folder_name + "/weight_mem_True_batch_64_trLRC_0.001_trainLR_0.0001_tuneLR_0.001_Mix_P_1_Epoch_20_Serv_ratio_0.2_Num_Class_10_same_Clients_1with_s_netTrues_netVGGc_netALEXNET.pkl"
         }



def get_client_accuracies(data_dict, all_iterations,c_to_remove):
    all_accuracies = {iteration: [] for iteration in all_iterations}
    get_all_accuracies(data_dict, all_accuracies, c_to_remove,all_iterations)
    return all_accuracies


class RunType(Enum):
    a_04125 = 1
    b_05125 = 2

if __name__ == '__main__':
    # Open and read the pickle file
    is_with_mem = True
    data_type_ = data_type_enum.non_iid
    max_iterations = 11
    run_type = RunType.b_05125
    if run_type ==RunType.a_04125:
        name_file_dict = get_name_file_dict_with_mem0401("04125", data_type_)
    if run_type == RunType.b_05125:
        name_file_dict = get_name_file_dict_with_mem0501("05125", data_type_)

    colors_dict = {"No server":"black","S.AlexNet":"blue","S.VGG16":"red"}

    v = "C_alex_S_vggNonIIDfull_2_KMeans_same_output_per_client_mean.pkl"
    with open(v, 'rb') as file:
        data_ = pickle.load(file)
    print()

    c_to_remove = []
    for k,v in name_file_dict.items():
        with open(v, 'rb') as file:
            data_ = pickle.load(file)


        data_dict = data_.accuracy_test_measures
        #data_dict = data_.accuracy_pl_measures
        all_iterations = get_all_iterations(data_dict,max_iterations)
        all_accuracies = get_client_accuracies(data_dict,all_iterations,c_to_remove)
        average_accuracies = [np.mean(all_accuracies[iteration]) for iteration in all_iterations]
        plt.plot(
            all_iterations, average_accuracies, label=k+"_clients", color=colors_dict[k], linestyle="solid", linewidth=2
        )

        server_ = get_server_data(data_dict,max_iterations)
        if len(server_)==len(all_iterations):
            plt.plot(
                all_iterations, server_, label=k+"_server", color=colors_dict[k], linestyle='--', linewidth=2
            )
    finalize_plot(data_type_.name)





        #plot_average_curve(all_accuracies, all_iterations, label_)

    #plot_client_accuracies(data_dict,[],11)#[7],11)