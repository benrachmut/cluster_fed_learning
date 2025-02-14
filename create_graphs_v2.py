# File path to your pickle file
import pickle
from itertools import cycle

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils.fixes import percentile

from main_ import *
from config import *

import numpy as np
import os


def extract_data():
    with open(file_name, 'rb') as file:
        return pickle.load(file)


class Measure (Enum):
    Local_Client_Validation = 1
    Global_Test_Data = 2


def get_iteration_list(client_data):
    for c_id,dict_ in client_data.items():
        return list(dict_.keys())

def handle_data_accuracy_per_client_1_max(client_data):
    iterations_list = get_iteration_list(client_data)
    ans = {}
    for i in iterations_list:
        list_per_i = []
        for dict_ in client_data.values():
            list_per_i.append(dict_[i])
        ans[i] = sum(list_per_i)/len(list_per_i)
    return ans


def get_dat_server_clients(cluster_amount):

    try:


        single_data = data_[type_data][data_type][amount_of_clients][str(percent_mix)][net_type][server_arch][cluster_amount][server_input_tech][cluster_tech][feedback]

        client_data = None
        server_data = None
        if measure == Measure.Local_Client_Validation:
            client_data = single_data.client_accuracy_per_client_1
            client_data = handle_data_accuracy_per_client_1_max(client_data)
            server_data = single_data.server_accuracy_per_client_1_max
            server_data = handle_data_accuracy_per_client_1_max(server_data)

        if measure == Measure.Global_Test_Data:
            client_data = single_data.client_accuracy_test_global_data
            client_data = handle_data_accuracy_per_client_1_max(client_data)
            server_data = single_data.server_accuracy_per_cluster_test_1
            server_data = handle_data_accuracy_per_client_1_max(server_data)


        return server_data, client_data
    except:
        return None,None
def get_ana_data(cluster_num_list):


    ans = {}
    for cluster_num in cluster_num_list:
        server_data, client_data = get_dat_server_clients(cluster_num)
        if server_data is None:
            break
        ans[cluster_num]= {}

        ans[cluster_num]["server"] = server_data
        ans[cluster_num]["clients"] = client_data
    return ans



    for cluster_amount in  cluster_num_list:
        ans[cluster_amount] = {}
        for server_input_tech in server_input_tech_list:
            ans[cluster_amount][server_input_tech] = {}
            for feedback in feedback_list:
                server_data, client_data = get_dat_server_clients (cluster_amount,feedback,server_input_tech)
                if server_data is None:
                    break
                ans[cluster_amount][server_input_tech][feedback] = {}

                ans[cluster_amount][server_input_tech][feedback]["server"] = server_data
                ans[cluster_amount][server_input_tech][feedback]["clients"] = client_data

    return ans

def twist_data():
    ans = {}
    for cluster_num, dict_1 in data_.items():
        for input_tech, dict_2 in dict_1.items():
            if input_tech not in ans:
                ans[input_tech] = {}
            for output_tech, dict_3 in dict_2.items():
                if output_tech not in ans[input_tech]:
                    ans[input_tech][output_tech]={}
                ans[input_tech][output_tech][cluster_num] = dict_3
    return ans




def create_graph():
    # Get the default color cycle from Matplotlib
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = cycle(default_colors)

    # Define a set of line styles
    line_styles = cycle(['-', '--', '-.', ':'])

    # Create a mapping from inner keys to line styles
    line_style_mapping = {}
    for outer_key, curves in data_for_graph.items():
        for inner_key in curves.keys():
            if inner_key not in line_style_mapping:
                line_style_mapping[inner_key] = next(line_styles)

    plt.figure(figsize=(10, 6))  # Adjust figure size for better legend placement

    # Plot each curve
    for outer_key, curves in data_for_graph.items():
        color = next(color_cycle)  # Assign a color for each outer_key
        for inner_key, points in curves.items():
            x = sorted(points.keys())
            y = [points[k] for k in x]
            plt.plot(x, y,
                     color=color,
                     linestyle=line_style_mapping[inner_key],
                     label=f"{inner_key} - {outer_key}")

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy Percentage_" + measure.name)
    plt.title(graph_name)
    plt.ylim(25,75)
    # Move the legend to the right outside the plot
    plt.legend(title="Entity and clusters #", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()  # Adjust layout to fit the legend better

    # Save or show the graph
    if create_jpeg:
        filename = f"{graph_name}_{measure.name}.jpeg"
        plt.savefig(filename, format='jpeg', bbox_inches="tight")  # Ensure the legend is included
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':


    create_jpeg=False
    file_name = "increase num_of clients cifar 10/CIFAR10_NonIID_50_20_C_alex_S_alex_5_multi_head_max_manual_similar_to_cluster.pkl"
    data_ = extract_data()
    type_data = "CIFAR10"
    data_type = "NonIID"
    amount_of_clients = 25
    percent_mix = 20
    net_type = "C_alex_S_alex"#C_alex_S_vgg
    server_arch = "multi_model"#"multi_head"#"multi_model"
    cluster_num_list = ["known_labels",1,3,5]
    server_input_tech = "max"
    cluster_tech = "manual"
    feedback = "similar_to_cluster"


    measure = Measure.Local_Client_Validation

    data_ = get_ana_data(cluster_num_list)
    #data_ = twist_data()
    first_part_graph_name = server_arch + "_" + net_type + "_" + str(percent_mix)


    graph_name = first_part_graph_name
    data_for_graph = data_
    create_graph()




