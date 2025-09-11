import os
import pickle
from pickle import FALSE

from jinja2.nodes import Break
from matplotlib.lines import Line2D

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
from typing import Dict, List, Iterable, Tuple, Union, Any
from collections import defaultdict

change_dict_name_server_client = {"MAPL,AlexNet":"AlexNet","MAPL,VGG":"VGG"}
net_name = {"C_alex_S_alex": "S_AlexNet", "C_alex_S_vgg": "S_VGG-16"}#
seeds_dict = {100:{DataSet.CIFAR100.name:[1,2,3],DataSet.CIFAR10.name:[2,4,5],DataSet.EMNIST_balanced.name:[1,2,3],DataSet.TinyImageNet.name:[1,2,3]}

#5:{DataSet.CIFAR100.name:[1,2,3,5,7],DataSet.CIFAR10.name:[2,4,5,6,9]
,5:{DataSet.CIFAR100.name:[1,2,3],DataSet.CIFAR10.name:[2,4,5],DataSet.EMNIST_balanced.name:[1,2,3],DataSet.TinyImageNet.name:[1,2,3]}
,1:{DataSet.CIFAR100.name:[1,2,3],DataSet.CIFAR10.name:[2,4,5],DataSet.EMNIST_balanced.name:[1,2,3],DataSet.TinyImageNet.name:[1,2,3]}}
colors = {"MAPL,VGG": "blue",
"COMET":"Orange",
          "MAPL,AlexNet": "red",
          "FedMd": "Green",
          "No FL": "Gray",
          "FedAvg": "brown",
          "pFedCK": "purple"
          }
Run = Dict[Any, Dict[Union[int, str], float]]  # {client_id: {iteration: accuracy}}

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


def update_data_v3(data, data_type, flag=False):
    """
    - If flag is False:
        * drop entries where x == 1
        * shift all x > 1 to x-1
        * keep x < 1 unchanged
    - If flag is True: leave keys as-is.
    Then add x=0.0 seeded with start_point[data_type] repeated to match the
    series length at x=1 (after shifting) or, if missing, any series length.
    """
    if not flag:
        new_xy = {}
        for x, y in data.items():
            # remove x == 1
            if x == 1 or x == 1.0:
                continue
            # shift x > 1 down by 1
            new_x = (x - 1) if (x > 1) else x
            new_xy[new_x] = y
    else:
        new_xy = dict(data)

    # Determine how many seeds to add at x=0.0
    # Prefer the length at x=1 (after shift), else fallback to any series length, else 0.
    any_series = next(iter(new_xy.values()), [])
    series_len = len(new_xy.get(1, any_series))

    new_xy[0.0] = [start_point[data_type]] * series_len
    return dict(sorted(new_xy.items()))

def update_data_v2(data, data_type, flag = False):
    #for algo, xy_dict in data.items():
    if not flag:
        new_xy = {x + 1: y for x, y in data.items()}  # shift x keys by +1
    else:
        new_xy = {x : y for x, y in data.items()}  # shift x keys by +1

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

def switch_algo_and_seedV2(merged_dict):
    rds = {}
    for seed in [1]:
        for algo in merged_dict[seed]:
            algo_name = algo_names[algo]
            #if algo == AlgorithmSelected.PseudoLabelsClusters.name:
            rds[algo_name] = []
                #algo_name_list = get_PseudoLabelsClusters_name(algo,merged_dict[seed][algo])
                #for name_ in algo_name_list:
                #    if name_ not in rds.keys() :
                #        rds[name_] = []

            #elif algo_name not in rds.keys() :
            #    rds[algo_name] = []
            rd_output = extract_rd(algo, merged_dict[seed][algo])
            if isinstance(rd_output,dict):
                for k,v in rd_output.items():
                    rds[algo_name].append(v)
                    break
            else:
                rds[algo_name].append(rd_output)



            #ans[algo].append(merged_dict[seed][algo])
    return rds

def switch_algo_and_seed(merged_dict,dich,data_type):
    rds = {}
    for seed in [1]:
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
                h = stderr * stats.norm.ppf((1 + confidence) / 2.)

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

    # Adjust layout to create more space for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Increase the bottom margin
    fig.subplots_adjust(top=0.8)  # Increase space between the plots and the legend

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
                h = stderr * stats.norm.ppf((1 + confidence) / 2.)

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
                    h = stderr * stats.norm.ppf((1 + confidence) / 2.)

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


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_model_algos_v2(
    data_dict,
    x_label="Iterations",
    y_label="Top-1 Accuracy (%)",
    confidence=0.95,
):
    """
    Expects:
      data_dict[alpha][algorithm][role]["iteration"] -> list of metric values
        where role is usually "Clients" and/or "Server".

    Behavior:
      - For algorithms whose name contains "MAPL" (case-insensitive), plot the
        "Server" series.
      - For all other algorithms, plot the "Clients" series.
      - Two subplots (one per alpha). Different algorithms = different colors.
      - Confidence intervals are shaded (normal-approx CI from SEM).
      - Shared Y-limits computed only from the actually plotted series
        (MAPL->Server; others->Clients).
    """
    assert len(data_dict) == 2, "Expected exactly 2 alphas for a 1x2 plot."

    import itertools
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    # ----- collect algorithms (stable, sorted) -----
    all_algos = set()
    for alpha_data in data_dict.values():
        for algo_name in alpha_data.keys():
            all_algos.add(algo_name)
    algo_list = sorted(all_algos)

    # color map for algorithms
    tab10 = plt.get_cmap("tab10").colors
    color_cycle = itertools.cycle(tab10)
    algo_colors = {algo: next(color_cycle) for algo in algo_list}

    def desired_role_for_algo(algo_name: str) -> str:
        return "Server" if "mapl" in algo_name.lower() else "Clients"

    # ----- compute global y-limits from *selected* roles only -----
    all_vals = []
    for alpha_data in data_dict.values():
        for algo_name in algo_list:
            role_key = desired_role_for_algo(algo_name)
            role_dict = alpha_data.get(algo_name, {}).get(role_key, None)
            if not isinstance(role_dict, dict):
                continue
            for vals in role_dict.values():
                all_vals.extend(vals)

    if not all_vals:
        raise ValueError('No values found in the selected roles (MAPL->"Server", others->"Clients").')

    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    pad = max(1e-6, 0.03 * (vmax - vmin) if vmax > vmin else 1.0)
    ymin = float(vmin - pad)
    ymax = float(vmax + pad)

    # ----- figure & alpha ordering -----
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs = axs.flatten()

    alpha_items = list(data_dict.items())
    try:
        alpha_items.sort(key=lambda kv: float(kv[0]))
    except Exception:
        pass

    global_lines, global_labels, seen_labels = [], [], set()

    for i, (alpha_name, alpha_data) in enumerate(alpha_items):
        ax = axs[i]

        for algo_name in algo_list:
            role_key = desired_role_for_algo(algo_name)
            role_dict = alpha_data.get(algo_name, {}).get(role_key, None)
            if not isinstance(role_dict, dict):
                # Nothing to plot for this (alpha, algo) with the chosen role.
                continue

            # x-values are iterations (keys) sorted numerically
            x_values = sorted(role_dict.keys(), key=lambda x: float(x))
            means, lower_bounds, upper_bounds = [], [], []

            for it in x_values:
                vals = np.asarray(role_dict[it], dtype=float)
                if vals.size == 0 or np.all(np.isnan(vals)):
                    means.append(np.nan)
                    lower_bounds.append(np.nan)
                    upper_bounds.append(np.nan)
                    continue

                mean = np.nanmean(vals)
                n = np.sum(~np.isnan(vals))
                if n <= 1:
                    h = 0.0
                else:
                    stderr = stats.sem(vals, nan_policy="omit")
                    h = stderr * stats.norm.ppf((1 + confidence) / 2.0)

                means.append(mean)
                lower_bounds.append(mean - h)
                upper_bounds.append(mean + h)

            x_values = np.array(x_values, dtype=float)
            means = np.array(means, dtype=float)
            lower_bounds = np.array(lower_bounds, dtype=float)
            upper_bounds = np.array(upper_bounds, dtype=float)

            mask = ~np.isnan(means)
            x_plot = x_values[mask]
            m_plot = means[mask]
            lb_plot = lower_bounds[mask]
            ub_plot = upper_bounds[mask]

            if x_plot.size == 0:
                continue

            color = algo_colors[algo_name]
            # Make it explicit in the legend when we used Server for MAPL
            label = f"{algo_name} (Server)" if role_key == "Server" else algo_name

            line, = ax.plot(x_plot, m_plot, label=label, linewidth=2.0, color=color)
            ax.fill_between(x_plot, lb_plot, ub_plot, alpha=0.2, color=color)

            if label not in seen_labels:
                global_lines.append(line)
                global_labels.append(label)
                seen_labels.add(label)

        ax.set_title(f"$\\alpha = {alpha_name}$", fontsize=axes_number_font)
        ax.set_xlabel(x_label, fontsize=axes_titles_font)
        ax.set_ylabel(y_label, fontsize=axes_titles_font)
        ax.tick_params(axis="both", labelsize=tick_font_size)
        ax.set_ylim(ymin, ymax)

    fig.legend(
        global_lines,
        global_labels,
        loc="upper center",
        ncol=max(1, len(global_labels)),
        fontsize=legend_font_size,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figures/algos_mixed_roles_alphas.pdf", format="pdf")
    plt.show()
    return fig

def plot_model_algos(data_dict, x_label="Iterations", y_label="Top-1 Accuracy (%)", confidence=0.95):
    """
    Expects:
      data_dict[alpha][algorithm]["Clients"][iteration] -> list of metric values

    Plots a 1x2 grid (two alphas). Each subplot shows the "Clients" curve for
    each algorithm; different algorithms = different line colors. Confidence
    intervals are shaded. Y-limits are shared across subplots based on the
    selected "Clients" data only.
    """
    assert len(data_dict) == 2, "Expected exactly 2 alphas for a 1x2 plot."

    # Build a stable algorithm list across all alphas (order by name for consistency)
    all_algos = set()
    for alpha_data in data_dict.values():
        for algo_name in alpha_data.keys():
            all_algos.add(algo_name)
    algo_list = sorted(all_algos)

    # Color map for algorithms (fallback to tab10 cycling)
    import itertools
    tab10 = plt.get_cmap("tab10").colors
    color_cycle = itertools.cycle(tab10)
    algo_colors = {algo: next(color_cycle) for algo in algo_list}

    # --- Compute global ymin/ymax from the "Clients" series only ---
    all_vals = []
    for alpha_data in data_dict.values():
        for algo_name, role_dict in alpha_data.items():
            if "Clients" not in role_dict:
                continue
            iter_dict = role_dict["Clients"]
            for vals in iter_dict.values():
                all_vals.extend(vals)

    if not all_vals:
        raise ValueError('No values found under key "Clients" to plot.')

    # Use a small pad for nicer aesthetics
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    pad = max(1e-6, 0.03 * (vmax - vmin) if vmax > vmin else 1.0)
    ymin = float(vmin - pad)
    ymax = float(vmax + pad)

    # --- Create figure and axes ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs = axs.flatten()

    global_lines = []
    global_labels = []
    seen_labels = set()

    # Ensure deterministic alpha panel order
    alpha_items = list(data_dict.items())
    # Sort by the alpha key if sortable; otherwise keep insertion order
    try:
        alpha_items.sort(key=lambda kv: float(kv[0]))
    except Exception:
        pass

    for i, (alpha_name, alpha_data) in enumerate(alpha_items):
        ax = axs[i]

        for algo_name in algo_list:
            role_dict = alpha_data.get(algo_name, {})
            if "Clients" not in role_dict:
                # Skip if this alpha lacks Clients data for the algorithm
                continue

            iter_dict = role_dict["Clients"]
            x_values = sorted(iter_dict.keys())

            means = []
            lower_bounds = []
            upper_bounds = []

            for it in x_values:
                vals = np.asarray(iter_dict[it], dtype=float)
                if vals.size == 0 or np.all(np.isnan(vals)):
                    means.append(np.nan)
                    lower_bounds.append(np.nan)
                    upper_bounds.append(np.nan)
                    continue

                mean = np.nanmean(vals)
                # Handle SEM safely (n could be 1)
                n = np.sum(~np.isnan(vals))
                if n <= 1:
                    h = 0.0
                else:
                    stderr = stats.sem(vals, nan_policy='omit')
                    h = stderr * stats.norm.ppf((1 + confidence) / 2.0)

                means.append(mean)
                lower_bounds.append(mean - h)
                upper_bounds.append(mean + h)

            x_values = np.array(x_values, dtype=float)
            means = np.array(means, dtype=float)
            lower_bounds = np.array(lower_bounds, dtype=float)
            upper_bounds = np.array(upper_bounds, dtype=float)

            # Drop NaNs if any (keeps lines clean)
            mask = ~np.isnan(means)
            x_plot = x_values[mask]
            m_plot = means[mask]
            lb_plot = lower_bounds[mask]
            ub_plot = upper_bounds[mask]

            color = algo_colors[algo_name]
            label = algo_name  # only algorithm name, since all are "Clients"

            if x_plot.size > 0:
                line, = ax.plot(x_plot, m_plot, label=label, linewidth=2.0, color=color)
                ax.fill_between(x_plot, lb_plot, ub_plot, alpha=0.2, color=color)

                if label not in seen_labels:
                    global_lines.append(line)
                    global_labels.append(label)
                    seen_labels.add(label)

        # Titles, labels, ticks
        ax.set_title(f"$\\alpha = {alpha_name}$", fontsize=axes_number_font)
        ax.set_xlabel(x_label, fontsize=axes_titles_font)
        ax.set_ylabel(y_label, fontsize=axes_titles_font)
        ax.tick_params(axis='both', labelsize=tick_font_size)
        ax.set_ylim(ymin, ymax)

    # Shared legend at the top
    fig.legend(global_lines, global_labels, loc='upper center',
               ncol=max(1, len(global_labels)), fontsize=legend_font_size, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figures/clients_only_alphas.pdf", format="pdf")
    plt.show()
    return fig


def plot_model_server_client(data_dict, x_label="Iterations", y_label="Top-1 Accuracy (%)", confidence=0.95):
    assert len(data_dict) == 1, "Expected exactly 1 alpha for this plot."

    line_styles = {"Server": "solid", "Clients": "dashed"}

    # Give the right side some room for side legends
    fig, ax = plt.subplots(figsize=(6.5, 5))  # a touch wider helps
    fig.subplots_adjust(right=0.78)           # reserve space on the right

    global_lines = []
    global_labels = []
    seen_labels = set()

    # Compute global ymin and ymax (kept as in your code; not enforced)
    all_vals = []
    for model_data in data_dict.values():
        for role_data in model_data.values():
            for iter_data in role_data.values():
                for vals in iter_data.values():
                    all_vals.extend(vals)
    xmin = 0
    xmax = 15

    # Plot each model-role curve
    for alpha_name, model_data in data_dict.items():
        for model_name, role_data in model_data.items():
            for role, iter_data in role_data.items():
                linestyle = line_styles.get(role, "solid")
                x_values = sorted(iter_data.keys())
                means, lower_bounds, upper_bounds = [], [], []

                for it in x_values:
                    vals = np.array(iter_data[it])
                    mean = np.mean(vals)
                    stderr = stats.sem(vals)
                    h = stderr * stats.norm.ppf((1 + confidence) / 2.0)
                    means.append(mean)
                    lower_bounds.append(mean - h)
                    upper_bounds.append(mean + h)

                x_values = np.array(x_values)
                means = np.array(means)
                lower_bounds = np.array(lower_bounds)
                upper_bounds = np.array(upper_bounds)

                label = f"{model_name}-{role}"
                (line,) = ax.plot(x_values, means, label=label, linestyle=linestyle)
                ax.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2)

                if label not in seen_labels:
                    global_lines.append(line)
                    global_labels.append(label)
                    seen_labels.add(label)

        ax.set_title(f"$\\alpha = {alpha_name}$", fontsize=axes_number_font)
        ax.set_xlabel(x_label, fontsize=axes_titles_font)
        ax.set_ylabel(y_label, fontsize=axes_titles_font)
        ax.tick_params(axis="both", labelsize=tick_font_size)


    # -------- Side legends --------
    # Main legend: all model-role curves, on the right side
    leg_models = fig.legend(
        global_lines,
        global_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=legend_font_size,
        title="Model–Role",
        title_fontsize=legend_font_size,
    )

    # Line-style legend: explains Server vs Clients
    role_handles = [
        Line2D([0], [0], linestyle=line_styles["Server"], linewidth=2, color="k"),
        Line2D([0], [0], linestyle=line_styles["Clients"], linewidth=2, color="k"),
    ]
    leg_styles = fig.legend(
        role_handles,
        ["Server", "Clients"],
        loc="upper left",
        bbox_to_anchor=(1.02, 0.95),
        frameon=False,
        fontsize=max(16 - 2, 8),
        title="Line Style",
        title_fontsize=max(16 - 2, 8),
    )
    # Ensure both legends render
    ax.add_artist(leg_models)
    ax.add_artist(leg_styles)
    ax.set_xlim(xmin, xmax)

    # --------------------------------


    return plt,fig

def get_PseudoLabelsClusters_name(algo,dict_):
    ans = []
    for net_type in dict_.keys():
        if net_type == NetsType.C_alex_S_vgg.name:
            algo_name = "C=AlexNet,S=VGG"
        if net_type == NetsType.C_alex_S_alex.name:
            algo_name = "C=AlexNet,S=AlexNet"
        if net_type == NetsType.C_MobileNet_S_vgg.name:
            algo_name = "C=MobileNet,S=VGG"
        if net_type == NetsType.C_rnd_S_alex.name:
            algo_name = "C=Random,S=AlexNet"
        if net_type == NetsType.C_rnd_S_Vgg.name:
            algo_name = "C=Random,S=VGG"
        if net_type == NetsType.C_MobileNet_S_alex.name:
            algo_name = "C=MobileNet,S=VGG"


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




        if name == "C=AlexNet,S=VGG":
            dict_1 = dict_[NetsType.C_alex_S_vgg.name]
        if name == "C=AlexNet,S=AlexNet":
            dict_1 = dict_[NetsType.C_alex_S_alex.name]
        if name == "C=MobileNet,S=VGG":
            dict_1 = dict_[NetsType.C_MobileNet_S_vgg.name]
        if name == "C=Random,S=VGG":
            dict_1 = dict_[NetsType.C_rnd_S_Vgg.name]

        if name == "C=Random,S=AlexNet":
            dict_1 = dict_[NetsType.C_rnd_S_alex.name]
        if name == "C=random,S=VGG":
            dict_1 = dict_[NetsType.C_rnd_S_alex.name]

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
        if algo == "MAPL":
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

            data_per_iteration_client = update_data_v3(data_per_iteration_client, data_type)
            data_per_iteration_server = update_data_v2(data_per_iteration_server, data_type, flag=True)

            #ans[algo] = data_per_iteration_client
            ans[algo]={"Clients":data_per_iteration_client}
            ans[algo]["Server"] = data_per_iteration_server
            print()
        else:
            data_per_iteration_client = {}
            for rd in rd_list:
                data_per_client, data_per_server = get_data_per_client_and_server(rd, algo, top_what, data_type)
                for client_id, data_dict in data_per_client.items():
                    for iter_, v in data_dict.items():
                        if iter_ not in data_per_iteration_client.keys():
                            data_per_iteration_client[iter_] = []
                        data_per_iteration_client[iter_].append(v)
                #############################################################


            data_per_iteration_client = update_data_v3(data_per_iteration_client, data_type)

            # ans[algo] = data_per_iteration_client
            ans[algo] = {"Clients": data_per_iteration_client}


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


def aggregate_all_clients_seeds(
    runs: List[Run],
    add_start_point: bool = True
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate across ALL clients and ALL seeds.
    Optionally force a synthetic start point (iter=0, acc=1.0).
    """
    bag = defaultdict(list)  # iter -> [acc values]

    for run in runs:
        if not isinstance(run, dict):
            continue
        for client, iter_to_acc in run.items():
            if not isinstance(iter_to_acc, dict):
                continue
            for it, acc in iter_to_acc.items():
                try:
                    it_i = int(it)
                    val = float(acc)
                except Exception:
                    continue
                if np.isfinite(val):
                    bag[it_i].append(val)

    # Force the start point to be exactly 1.0 (as in your code)
    if add_start_point:
        bag[0] = [1.0]   # replaces any existing iter=0 values

    if not bag:
        return [], np.array([]), np.array([]), np.array([])

    iterations = sorted(bag.keys())
    means  = np.array([np.mean(bag[it]) for it in iterations], dtype=float)
    stds   = np.array([np.std(bag[it], ddof=1) if len(bag[it]) > 1 else 0.0 for it in iterations], dtype=float)
    counts = np.array([len(bag[it]) for it in iterations], dtype=int)
    return iterations, means, stds, counts

