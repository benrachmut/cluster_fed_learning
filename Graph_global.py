import os
import pickle

from config import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"MAPL"
        ,AlgorithmSelected.PseudoLabelsNoServerModel.name:"FedMd",
        AlgorithmSelected.NoFederatedLearning.name:"No FL",
        AlgorithmSelected.Centralized.name:"Centralized",
                AlgorithmSelected.FedAvg.name:"FedAvg",
                AlgorithmSelected.pFedCK.name:"pFedCK",
            AlgorithmSelected.COMET.name: "COMET"

            }
#1,2,3,5,7

change_dict_name_server_client = {"MAPL,AlexNet":"AlexNet","MAPL,VGG":"VGG"}
net_name = {"C_alex_S_alex": "S_AlexNet", "C_alex_S_vgg": "S_VGG-16"}#
seeds_dict = {100:{DataSet.CIFAR100.name:[1],DataSet.CIFAR10.name:[2],DataSet.EMNIST_balanced.name:[1],DataSet.TinyImageNet.name:[1]}

#5:{DataSet.CIFAR100.name:[1,2,3,5,7],DataSet.CIFAR10.name:[2,4,5,6,9]
,5:{DataSet.CIFAR100.name:[1,2,3],DataSet.CIFAR10.name:[2,4,5],DataSet.EMNIST_balanced.name:[1,2,3],DataSet.TinyImageNet.name:[1,2,3]}}
colors = {"MAPL,VGG": "blue",
"COMET":"Orange",
          "MAPL,AlexNet": "red",
          "FedMd": "Green",
          "No FL": "Gray",
          "FedAvg": "brown",
          "pFedCK": "purple"
          }

def read_all_pkls(folder_path):
    # Path to the folder containing pickle files


    # List to store the data from each pickle file
    all_data = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a pickle file (ends with .pkl)
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)

            # Open the pickle file and load the data
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                all_data.append(data)
    return all_data

def merge_dicts(dict_list):
    """
    Merges a list of dictionaries recursively. It combines the dictionaries in the list
    by updating the values of matching keys and merging nested dictionaries.
    """
    combined = {}
    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key in combined:
                if isinstance(combined[key], dict) and isinstance(value, dict):
                    # If both are dictionaries, merge them recursively
                    combined[key] = merge_dicts([combined[key], value])
                else:
                    # You can customize how to merge non-dict values
                    combined[key] = value  # Update with the new value
            else:
                combined[key] = value  # Add new key-value pair
    return combined

axes_titles_font = 16
axes_number_font = 14
legend_font_size = 16
tick_font_size = 14
linewidth = 3


def create_algo_cluster(data, x_label, y_label, folder_to_save, figure_name,y_lim = None):
    linewidth = 2
    markersize = 5

    # Create the main figure
    fig, ax = plt.subplots(figsize=(4, 3))

    x_values = list(data.keys())
    y_values = list(data.values())

    ax.plot(
        x_values,
        y_values,
        marker='o',
        linewidth=linewidth,
        linestyle='solid',
        markersize=markersize
    )

    # Set axis labels
    ax.set_xlabel(x_label, fontsize=axes_titles_font)
    ax.set_ylabel(y_label, fontsize=axes_titles_font)

    # Tick font size
    ax.tick_params(axis='both', labelsize=tick_font_size)

    # Optional: Set fixed y-limits
    #ax.set_ylim([12, 38])

    # Save the figure
    fig.savefig(f"{folder_to_save}/{figure_name}.pdf", format="pdf", bbox_inches='tight')
    plt.close(fig)
start_point = {DataSet.CIFAR100.name:1,DataSet.CIFAR10.name:10,DataSet.TinyImageNet.name:0.5,DataSet.EMNIST_balanced.name:2.13,DataSet.SVHN.name:10}

def update_data(data, data_type):
    for algo, xy_dict in data.items():
        new_xy = {x + 1: y for x, y in xy_dict.items()}  # shift x keys by +1
        new_xy[0.0] = start_point[data_type]             # add new point at x = 0
        data[algo] = dict(sorted(new_xy.items()))        # optional: sort by x if desired

def update_data_v2(data, data_type):
    #for algo, xy_dict in data.items():
    new_xy = {x + 1: y for x, y in data.items()}  # shift x keys by +1
    new_xy[0.0] = []
    for _ in range(0,len(new_xy[1])):
        new_xy[0.0].append(start_point[data_type])             # add new point at x = 0
    data = dict(sorted(new_xy.items()))
    return data
def create_algo_graph(data, x_label, y_label, folder_to_save, figure_name,y_lim = None,confidence = 0.95):
    linewidth = 2
    markersize = 3

    # Create main figure
    fig, ax = plt.subplots(figsize=(4, 3))

    lines = []
    labels = []
    for algorithm_name, iter_dict in data.items():
        # Get iterations (x-values)
        x_values = sorted(iter_dict.keys())
        means = []
        lower_bounds = []
        upper_bounds = []

        for it in x_values:
            accs = np.array(iter_dict[it])  # List of accuracies for the current iteration
            mean = np.mean(accs)  # Mean accuracy
            stderr = stats.sem(accs)  # Standard error
            n = len(accs)  # Number of samples
            h = stderr * stats.t.ppf((1 + confidence) / 2., n - 1)  # Confidence interval half-width

            means.append(mean)
            lower_bounds.append(mean - h)
            upper_bounds.append(mean + h)
            # Convert to arrays for plotting
        x_values = np.array(x_values)
        means = np.array(means)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        # Get the color for the algorithm
        color = colors.get(algorithm_name, "black")

        # Plot the mean accuracy line
        line, = ax.plot(
            x_values,
            means,
            marker='o',
            linewidth=linewidth,
            color=color,
            linestyle='solid',
            markersize=markersize
        )
        lines.append(line)
        labels.append(algorithm_name)

        # Fill the confidence interval
        ax.fill_between(
            x_values,
            lower_bounds,
            upper_bounds,
            color=color,
            alpha=0.2
        )

    # Set axis labels


    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Optional: Set fixed y-limits
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # Tick font size
    ax.tick_params(axis='both', labelsize=12)

    # Show the main figure
    plt.show()

    # Save the main figure without legend
    fig.savefig(f"{folder_to_save}/{figure_name}.pdf", format="pdf", bbox_inches='tight')
    plt.close(fig)

    # Create a separate legend-only figure
    legend_fig, legend_ax = plt.subplots(figsize=(len(lines) * 1.5, 0.6))
    legend_ax.axis("off")

    legend = legend_ax.legend(
        lines,
        labels,
        fontsize=12,
        loc="center",
        ncol=len(lines),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.0
    )
    # Save legend figure
    legend_fig.savefig(f"{folder_to_save}/{figure_name}_legend.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats




def switch_algo_and_seed_cluster(merged_dict,dich,data_type):
    rds = {}
    for seed in seeds_dict[dich][data_type]:
        temp_dict = merged_dict[seed][AlgorithmSelected.PseudoLabelsClusters.name][NetsType.C_alex_S_vgg.name]["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"]
        for cluster_num,d2 in temp_dict.items():
            cluster_correct_num = cluster_num+5
            rd = d2[WeightForPS.withWeights.name][InputConsistency.withInputConsistency.name]
            if cluster_correct_num not in rds:
                rds[cluster_correct_num] = []
            rds[cluster_correct_num].append(rd)
    return rds

def switch_algo_and_seed(merged_dict,dich,data_type):
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


def create_1x4_algo_grid(all_data_dict, x_label, y_label_dict, y_lim_dict=None, confidence=0.95, dich=5):
    assert len(all_data_dict) == 4, "You must provide exactly 4 datasets for a 1x4 grid."

    linewidth = 2
    markersize = 3

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Changed from 2x2 to 1x4
    axs = axs.flatten()

    global_lines = []
    global_labels = []
    already_plotted_algorithms = set()

    # Manual order of algorithms in the legend
    manual_legend_order = ["MAPL,VGG", "COMET", "pFedCK", "FedMd", "FedAvg", "No FL"]

    for i, (title, data) in enumerate(all_data_dict.items()):
        ax = axs[i]
        for algorithm_name, iter_dict in data.items():
            x_values = sorted(iter_dict.keys())
            means = []
            lower_bounds = []
            upper_bounds = []

            for it in x_values:
                accs = np.array(iter_dict[it])
                mean = np.mean(accs)
                stderr = stats.sem(accs)
                n = len(accs)
                h = stderr * stats.t.ppf((1 + confidence) / 2., n - 1)

                means.append(mean)
                lower_bounds.append(mean - h)
                upper_bounds.append(mean + h)

            x_values = np.array(x_values)
            means = np.array(means)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)

            color = colors.get(algorithm_name, "black")

            line, = ax.plot(
                x_values,
                means,
                marker='o',
                linewidth=linewidth,
                color=color,
                linestyle='solid',
                markersize=markersize,
                label=algorithm_name if algorithm_name not in already_plotted_algorithms else None
            )

            if algorithm_name not in already_plotted_algorithms:
                global_lines.append(line)
                global_labels.append(algorithm_name)
                already_plotted_algorithms.add(algorithm_name)

            ax.fill_between(x_values, lower_bounds, upper_bounds, color=color, alpha=0.2)

        ax.set_title(title, fontsize=axes_number_font)
        ax.set_xlabel(x_label, fontsize=axes_titles_font)
        ax.set_ylabel(y_label_dict.get(title, "Metric"), fontsize=axes_titles_font)

        if y_lim_dict and title in y_lim_dict:
            ax.set_ylim(y_lim_dict[title])

        ax.tick_params(axis='both', labelsize=tick_font_size)

    # Reorder the lines and labels manually as per the desired order
    sorted_lines = []
    sorted_labels = []
    for label in manual_legend_order:
        if label in global_labels:
            idx = global_labels.index(label)
            sorted_lines.append(global_lines[idx])
            sorted_labels.append(global_labels[idx])

    global_lines = sorted_lines
    global_labels = sorted_labels

    # Create the legend with the manually ordered lines and labels
    fig.legend(global_lines, global_labels, loc='upper center', ncol=len(global_labels), fontsize=20, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjusted for wider layout
    fig.savefig("figures/all_algos_alpha_" + str(dich) + ".pdf", format="pdf")
    plt.show()
    return fig

def create_2x2_algo_grid(all_data_dict, x_label, y_label_dict, y_lim_dict=None, confidence=0.95, dich=5):
    assert len(all_data_dict) == 4, "You must provide exactly 4 datasets for a 2x2 grid."

    linewidth = 2
    markersize = 3

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    global_lines = []
    global_labels = []
    already_plotted_algorithms = set()

    # Manual order of algorithms in the legend
    manual_legend_order = [ "MAPL,VGG", "COMET","pFedCK", "FedMd","FedAvg", "No FL"]

    for i, (title, data) in enumerate(all_data_dict.items()):
        ax = axs[i]
        for algorithm_name, iter_dict in data.items():
            x_values = sorted(iter_dict.keys())
            means = []
            lower_bounds = []
            upper_bounds = []

            for it in x_values:
                accs = np.array(iter_dict[it])
                mean = np.mean(accs)
                stderr = stats.sem(accs)
                n = len(accs)
                h = stderr * stats.t.ppf((1 + confidence) / 2., n - 1)

                means.append(mean)
                lower_bounds.append(mean - h)
                upper_bounds.append(mean + h)

            x_values = np.array(x_values)
            means = np.array(means)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)

            color = colors.get(algorithm_name, "black")

            line, = ax.plot(
                x_values,
                means,
                marker='o',
                linewidth=linewidth,
                color=color,
                linestyle='solid',
                markersize=markersize,
                label=algorithm_name if algorithm_name not in already_plotted_algorithms else None
            )
            if algorithm_name not in already_plotted_algorithms:
                global_lines.append(line)
                global_labels.append(algorithm_name)
                already_plotted_algorithms.add(algorithm_name)

            ax.fill_between(x_values, lower_bounds, upper_bounds, color=color, alpha=0.2)

        ax.set_title(title, fontsize=axes_number_font)
        ax.set_xlabel(x_label, fontsize=axes_titles_font)
        ax.set_ylabel(y_label_dict.get(title, "Metric"), fontsize=axes_titles_font)

        if y_lim_dict and title in y_lim_dict:
            ax.set_ylim(y_lim_dict[title])

        ax.tick_params(axis='both', labelsize=tick_font_size)

    # Reorder the lines and labels manually as per the desired order
    sorted_lines = []
    sorted_labels = []
    for label in manual_legend_order:
        if label in global_labels:
            idx = global_labels.index(label)
            sorted_lines.append(global_lines[idx])
            sorted_labels.append(global_labels[idx])

    # Now update global_lines and global_labels to reflect the manual order
    global_lines = sorted_lines
    global_labels = sorted_labels

    # Create the legend with the manually ordered lines and labels
    fig.legend(global_lines, global_labels, loc='upper center', ncol=len(global_labels), fontsize=14, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("figures/all_algos_alpha_" + str(dich) + ".pdf", format="pdf")
    plt.show()
    return fig


def create_single_avg_plot(data_dict, x_label="X", y_label="Y", confidence=0.95, output_path="figures/number_of_clusters.pdf"):
    """
    Plots a blue average curve with confidence intervals from a dictionary.

    Parameters:
        data_dict (dict): Keys are x-values, values are lists of y-values.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        confidence (float): Confidence level for shading (default is 0.95).
        output_path (str): Path to save the figure as a PDF.
    """
    # Sort the data by x-values
    sorted_items = sorted(data_dict.items())
    x_vals = np.array([x for x, _ in sorted_items])
    means = []
    lower_bounds = []
    upper_bounds = []

    for _, values in sorted_items:
        values = np.array(values)
        mean = np.mean(values)
        stderr = stats.sem(values)
        n = len(values)
        h = stderr * stats.t.ppf((1 + confidence) / 2., n - 1) if n > 1 else 0

        means.append(mean)
        lower_bounds.append(mean - h)
        upper_bounds.append(mean + h)

    means = np.array(means)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    fig, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot(x_vals, means, color='blue', marker='o', linewidth=2, markersize=4, label="MAPL,VGG")
    ax.fill_between(x_vals, lower_bounds, upper_bounds, color='blue', alpha=0.2)

    ax.set_xlabel(x_label, fontsize=axes_titles_font)
    ax.set_ylabel(y_label, fontsize=axes_titles_font)
    ax.tick_params(axis='both', labelsize=tick_font_size)

    # Legend above the plot, centered
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, fontsize=legend_font_size, frameon=False)

    # Adjust layout to make space for legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    plt.show()
def plot_model_server_client_grid(data_dict, x_label="Iterations", y_label="Top-1 Accuracy (%)", confidence=0.95):
    assert len(data_dict) == 2, "Expected exactly 2 alphas for a 1x2 plot."

    model_colors = {"AlexNet": "red", "VGG": "blue"}
    line_styles = {"Server": "solid", "Clients": "dashed"}



    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs = axs.flatten()

    global_lines = []
    global_labels = []
    seen_labels = set()

    # First, compute global ymin and ymax
    all_vals = []
    for model_data in data_dict.values():
        for role_data in model_data.values():
            for iter_data in role_data.values():
                for vals in iter_data.values():
                    all_vals.extend(vals)

    ymin = 20
    ymax = 36

    for i, (alpha_name, model_data) in enumerate(data_dict.items()):
        ax = axs[i]
        for model_name, role_data in model_data.items():
            color = model_colors.get(model_name, "black")

            for role, iter_data in role_data.items():
                linestyle = line_styles.get(role, "solid")
                x_values = sorted(iter_data.keys())
                means = []
                lower_bounds = []
                upper_bounds = []

                for it in x_values:
                    vals = np.array(iter_data[it])
                    mean = np.mean(vals)
                    stderr = stats.sem(vals)
                    n = len(vals)
                    h = stderr * stats.t.ppf((1 + confidence) / 2., n - 1)

                    means.append(mean)
                    lower_bounds.append(mean - h)
                    upper_bounds.append(mean + h)

                x_values = np.array(x_values)
                means = np.array(means)
                lower_bounds = np.array(lower_bounds)
                upper_bounds = np.array(upper_bounds)

                label = f"{model_name}-{role}"
                line, = ax.plot(x_values, means, label=label, color=color, linestyle=linestyle)
                ax.fill_between(x_values, lower_bounds, upper_bounds, color=color, alpha=0.2, linestyle=linestyle)

                if label not in seen_labels:
                    global_lines.append(line)
                    global_labels.append(label)
                    seen_labels.add(label)

        ax.set_title(f"$\\alpha = {alpha_name}$", fontsize=axes_number_font)
        ax.set_xlabel(x_label, fontsize=axes_titles_font)
        ax.set_ylabel(y_label, fontsize=axes_titles_font)
        ax.tick_params(axis='both', labelsize=tick_font_size)
        ax.set_ylim(ymin, ymax)  # <-- Force same y-limits

    fig.legend(global_lines, global_labels, loc='upper center', ncol=len(global_labels), fontsize=legend_font_size, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figures/client_server_alphas.pdf", format="pdf")

    plt.show()
    return fig

def get_PseudoLabelsClusters_name(algo,dict_):
    ans = []
    for net_type in dict_.keys():
        algo_name = algo_names[algo] + ","
        if net_type == NetsType.C_alex_S_vgg.name:
            algo_name = algo_name + "VGG"
        if net_type == NetsType.C_alex_S_alex.name:
            algo_name = algo_name + "AlexNet"
        ans.append(algo_name)
    return ans


def get_data_per_client_client(rd, top_what):
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


def get_data_per_client_server(rd,top_what):
    if top_what == 1:
        return rd.server_accuracy_per_client_1_max
    if top_what == 5:
        return  rd.server_accuracy_per_client_5_max
    if top_what == 10:
        return  rd.server_accuracy_per_client_10_max

def get_data_per_client_and_server(rd,algo,top_what,data_type):
    data_per_server = get_data_per_client_server(rd, top_what)
    data_per_client = get_data_per_client_client(rd, top_what)
    return data_per_client,data_per_server
def get_data_per_client(rd,algo,top_what,data_type):
    if algo == algo_names[AlgorithmSelected.NoFederatedLearning.name]:
        data_per_client = get_data_per_client_client(rd,top_what)
        data_per_client = fix_data_NoFederatedLearning(data_per_client)
    if algo == algo_names[AlgorithmSelected.pFedCK.name] or algo == algo_names[
        AlgorithmSelected.PseudoLabelsNoServerModel.name]  or algo == algo_names[AlgorithmSelected.FedAvg.name]  or algo == algo_names[AlgorithmSelected.COMET.name]:
        data_per_client = get_data_per_client_client(rd,top_what)
    if algo ==algo_names[AlgorithmSelected.PseudoLabelsClusters.name]+",VGG" or algo ==algo_names[AlgorithmSelected.PseudoLabelsClusters.name]+",AlexNet":
        data_per_client = get_data_per_client_server(rd,top_what)
    update_data(data_per_client, data_type)
    return data_per_client



def extract_rd_FedAvg(algo,dict_):
    try:

        return dict_[NetsType.C_alex_S_alex.name]["multi_model"]["mean"]["kmeans"]["similar_to_cluster"][1]
    except:
        return dict_[NetsType.C_alex_S_alex.name]["no_model"]["mean"]["kmeans"]["similar_to_cluster"][1]



def extract_rd_pFedCK(algo,dict_):
    for net_type in dict_.keys():
        return dict_[net_type]

def extract_rd_PseudoLabelsClusters_server_client(algo,dict_):
    ans = {}

    names = get_PseudoLabelsClusters_name(algo, dict_)

    for name in names:
        if name == algo_names[algo]+",AlexNet":
            dict_1 = dict_[NetsType.C_alex_S_alex.name]
        if name == algo_names[algo]+",VGG":
            dict_1 = dict_[NetsType.C_alex_S_vgg.name]
        rd = dict_1["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"][0][
                WeightForPS.withWeights.name][InputConsistency.withInputConsistency.name]

        if name == "MAPL,VGG":
            name_to_place = "VGG"
        else:
            name_to_place = "AlexNet"
        ans[name_to_place] = rd

    return ans



def extract_rd_PseudoLabelsClusters(algo,dict_):
    ans = {}

    names = get_PseudoLabelsClusters_name(algo, dict_)

    for name in names:
        if name == algo_names[algo]+",AlexNet":
            dict_1 = dict_[NetsType.C_alex_S_alex.name]
        if name == algo_names[algo]+",VGG":
            dict_1 = dict_[NetsType.C_alex_S_vgg.name]

        try:
            rd = dict_1["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"][0][
                    WeightForPS.withWeights.name][InputConsistency.withInputConsistency.name]
        except:
            rd = dict_1["multi_model"]["max"][ClusterTechnique.greedy_elimination_L2.name]["similar_to_cluster"][
                WeightForPS.withWeights.name][InputConsistency.withInputConsistency.name][0]

        ans[name] = rd

    return ans

def extract_rd_PseudoLabelsNoServerModel(algo,dict_):

    return dict_["C_alex"]["no_model"]["mean"]["kmeans"]["similar_to_client"][1]




def extract_rd(algo,dict_):
    if algo == AlgorithmSelected.PseudoLabelsClusters.name:
        return extract_rd_PseudoLabelsClusters(algo,dict_)
    if algo == AlgorithmSelected.PseudoLabelsNoServerModel.name:
        return extract_rd_PseudoLabelsNoServerModel(algo,dict_)
    if algo == AlgorithmSelected.COMET.name :
        try:
            return  dict_[NetsType.C_alex_S_vgg.name]["multi_model"]["max"]["kmeans"]["similar_to_client"][5]
        except: return dict_["C_alex"]["no_model"]["mean"]["kmeans"]["similar_to_client"][5]
    if algo == AlgorithmSelected.NoFederatedLearning.name:
        return dict_["C_alex_S_alex"]
    if algo == AlgorithmSelected.FedAvg.name:
        return extract_rd_FedAvg(algo,dict_)
    if algo == AlgorithmSelected.pFedCK.name:
        return extract_rd_pFedCK(algo,dict_)

def collect_data_per_server_client_iteration(merged_dict,top_what,data_type):
    ans = {}
    for algo, rd_list in merged_dict.items():
        data_per_iteration_client = {}
        data_per_iteration_server = {}

        for rd in rd_list:
            data_per_client,data_per_server = get_data_per_client_and_server(rd,algo,top_what,data_type)
            for client_id, data_dict in data_per_client.items():
                for iter_, v in data_dict.items():
                    if iter_ not in data_per_iteration_client.keys():
                        data_per_iteration_client[iter_] = []
                    data_per_iteration_client[iter_].append(v)
            #############################################################
            for client_id, data_dict in data_per_server.items():
                for iter_, v in data_dict.items():
                    if iter_ not in data_per_iteration_server.keys():
                        data_per_iteration_server[iter_] = []
                    data_per_iteration_server[iter_].append(v)

        data_per_iteration_client = update_data_v2(data_per_iteration_client, data_type)
        data_per_iteration_server = update_data_v2(data_per_iteration_server, data_type)

        #ans[algo] = data_per_iteration_client
        ans[algo]={"Clients":data_per_iteration_client}
        ans[algo]["Server"] = data_per_iteration_server
        print()

    return ans

def collect_data_per_iteration(merged_dict,top_what,data_type):
    ans = {}
    for algo, rd_list in merged_dict.items():
        data_per_iteration = {}
        for rd in rd_list:
            data_per_client = get_data_per_client(rd,algo,top_what,data_type)
            for client_id, data_dict in data_per_client.items():
                for iter_, v in data_dict.items():
                    if iter_ not in data_per_iteration.keys():
                        data_per_iteration[iter_] = []
                    data_per_iteration[iter_].append(v)
        ans[algo] = data_per_iteration
    return ans

def create_variant_graph(data, x_label, y_label, folder_to_save, figure_name):
    # Define the font sizes and line width
    linewidth = 2
    line_styles = {
        "Clients": "dotted",
        "Server": "solid"
    }
    colors = {
        "w_i,w-IC": "tab:blue",
        "w_i,w/o-IC": "tab:red",
        "w_i=1,w/o-IC": "tab:green",
        "w_i=1,w-IC": "tab:brown",
    }

    # Create main figure without legend
    fig, ax = plt.subplots(figsize=(4, 3))

    solid_lines = []
    solid_labels = []
    dotted_lines = []
    dotted_labels = []

    for comm_type, models in data.items():
        for model_name, xy_values in models.items():
            x_values = list(xy_values.keys())
            y_values = list(xy_values.values())
            line, = ax.plot(
                x_values,
                y_values,
                marker='o',
                linewidth=linewidth,
                linestyle=line_styles.get(comm_type, "solid"),
                color=colors.get(model_name, "black"),
                markersize=3
            )
            if comm_type == "Server":
                solid_lines.append(line)
                solid_labels.append(f"{comm_type}-{model_name}")
            else:
                dotted_lines.append(line)
                dotted_labels.append(f"{comm_type}-{model_name}")

    # Set labels and ticks
    ax.set_xlabel(x_label, fontsize=axes_titles_font)
    ax.set_ylabel(y_label, fontsize=axes_titles_font)
    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.set_ylim([12, 38])

    # Save main figure (no legend)
    fig.savefig(f"{folder_to_save}/{figure_name}.pdf", format="pdf", bbox_inches='tight')
    plt.close(fig)

    # Create standalone legend figure
    total_lines = solid_lines + dotted_lines
    total_labels = solid_labels + dotted_labels

    legend_fig, legend_ax = plt.subplots(figsize=(max(4, len(total_lines) * 1.5), 1.2))  # Adjust size
    legend_ax.axis("off")

    # Create legend in two rows: first row solid, second row dotted
    legend = legend_ax.legend(
        total_lines,
        total_labels,
        fontsize=legend_font_size,
        loc="center",
        ncol=max(len(solid_lines), len(dotted_lines)),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.0,
        handletextpad=0.5
    )

    # Manually adjust spacing: first half solid, second half dotted
    for idx, handle in enumerate(legend.legendHandles):
        if idx >= len(solid_lines):
            handle.set_linestyle('dotted')

    # Save the legend figure
    legend_fig.savefig(f"{folder_to_save}/{figure_name}_legend.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)
    plt.close(legend_fig)

def create_CPL_graph(data, x_label, y_label, folder_to_save, figure_name):
    # Define the font sizes and line width
    linewidth = 2
    line_styles = {
        "Clients": "dotted",
        "Server": "solid"
    }
    colors = {
        "VGG": "tab:blue",
        "AlexNet": "tab:red"
    }

    # Create main figure without a legend
    fig, ax = plt.subplots(figsize=(4, 3))

    # Store legend elements
    lines = []
    labels = []

    for comm_type, models in data.items():  # client/server
        for model_name, xy_values in models.items():  # VGG/AlexNet
            x_values = list(xy_values.keys())
            y_values = list(xy_values.values())
            line, = ax.plot(
                x_values,
                y_values,
                marker='o',
                linewidth=linewidth,
                linestyle=line_styles.get(comm_type, "solid"),
                color=colors.get(model_name, "black"),
                markersize=3  # ← Set your desired marker size here

            )
            lines.append(line)
            labels.append(f"{comm_type}-{model_name}")

    # Set the labels with the specified font size
    ax.set_xlabel(x_label, fontsize=axes_titles_font)
    ax.set_ylabel(y_label, fontsize=axes_titles_font)

    # Set tick font size
    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.set_ylim([1, 38])

    # Save the main figure (without legend)
    fig.savefig(f"{folder_to_save}/{figure_name}.pdf", format="pdf", bbox_inches='tight')
    #fig.savefig(f"{folder_to_save}/{figure_name}.jpeg", format="jpeg", bbox_inches='tight')

    plt.close(fig)

    # Create standalone legend figure with a horizontal layout
    legend_fig, legend_ax = plt.subplots(figsize=(len(lines) * 1.5, 0.6))  # Adjust width dynamically
    legend_ax.axis("off")  # Hide axes

    # Create the legend
    legend = legend_ax.legend(
        lines,
        labels,
        fontsize=legend_font_size,
        loc="center",
        ncol=len(lines),
        frameon=False,
        handlelength=2.5,  # space between marker and label
        columnspacing=1.0  # tighter spacing between columns
    )
# Save with minimal padding
    #legend_fig.savefig(f"{folder_to_save}/{figure_name}_legend.jpeg", format="jpeg", bbox_inches='tight', pad_inches=0.05)
    legend_fig.savefig(f"{folder_to_save}/{figure_name}_legend.pdf", format="pdf", bbox_inches='tight', pad_inches=0.05)

def get_avg_of_entity(data_):
    lst_per_iter = {}
    for entity_id, dict_ in data_.items():
        for i, acc in dict_.items():
            if i not in lst_per_iter:
                lst_per_iter[i] = []
            lst_per_iter[i].append(acc)

    ans = {}
    for i, lst in lst_per_iter.items():
        ans[i] = sum(lst) / len(lst)
    return ans