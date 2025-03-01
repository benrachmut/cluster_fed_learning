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