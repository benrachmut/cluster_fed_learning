# File path to your pickle file
import pickle
from main_ import *
from config import *

import numpy as np

file_path = 'num_clusters_1_Mix_Percentage_0.2_Epochs_10_Iterations_50_Server_Split_Ratio_0.2_Num_Classes_2_Identical_Clients_2with_server_netFalse.pkl'






def get_all_iterations(data_dict):

    all_iterations = set()
    for client_data in data_dict.values():
        all_iterations.update(client_data.keys())
    return sorted(all_iterations)

def plot_individual_clients(data_dict, all_accuracies):
    """
    Plots the accuracy curves for individual clients and populates data for averaging.

    Args:
        data_dict (dict): Client accuracy data.
        all_accuracies (dict): Dictionary to store accuracies for averaging.
    """
    for client_id, accuracy_dict in data_dict.items():
        if isinstance(client_id,int):
            iterations = sorted(accuracy_dict.keys())
            accuracies = [accuracy_dict[iteration] for iteration in iterations]

            # Add accuracies to all_accuracies for averaging
            for iteration, acc in zip(iterations, accuracies):
                all_accuracies[iteration].append(acc)

            # Plot the individual client's curve
            #plt.plot(
            #    iterations, accuracies, label=f"Client {client_id}", marker='o', linestyle='-'
            #)

def plot_average_curve(all_accuracies, all_iterations):
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
        all_iterations, average_accuracies, label="Average Accuracy", color='black', linestyle='--', linewidth=2
    )

def finalize_plot():
    """
    Adds labels, legend, grid, and displays the plot.
    """
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Client Accuracies and Average")
    plt.legend(title="Clients")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_client_accuracies(data_dict):
    """
    Plots the accuracy of clients over iterations and the average accuracy curve.

    Args:
        data_dict (dict): A dictionary where:
                          - Key: Client ID (e.g., int or str)
                          - Value: Dictionary with key as iteration (int) and value as accuracy (float)
    """
    all_iterations = get_all_iterations(data_dict)
    all_accuracies =  {iteration: [] for iteration in all_iterations}

    plot_individual_clients(data_dict, all_accuracies)
    plot_average_curve(all_accuracies, all_iterations)

    finalize_plot()

if __name__ == '__main__':
    # Open and read the pickle file
    with open(file_path, 'rb') as file:
        data_ = pickle.load(file)
    print()
    #data_dict = data_.loss_measures
    data_dict = data_.accuracy_pl_measures
    #data_dict = data_.accuracy_test_measures

    plot_client_accuracies(data_dict)
