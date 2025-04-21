import os
import pickle

from config import *
from main_ import RecordData



algo_names={AlgorithmSelected.PseudoLabelsClusters.name:"C-PL",AlgorithmSelected.PseudoLabelsNoServerModel.name:"C-PL-NSM",AlgorithmSelected.NoFederatedLearning.name:"No FL"  }
net_name = {"C_alex_S_alex": "S_AlexNet", "C_alex_S_vgg": "S_VGG-16"}



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

axes_titles_font = 14
axes_number_font = 14
legend_font_size = 8
tick_font_size = 10
linewidth = 3


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
    ax.set_ylim([12, 38])

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