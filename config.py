from enum import Enum

from matplotlib import pyplot as plt


class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

iterations = 50
num_clients = 2
percent_train_data_use = 1
percent_test_relative_to_train = 1
client_split_ratio_list = [0.8,0.2,0.1,0.5]
server_net_type = NetType.VGG
client_net_type = NetType.ALEXNET
num_classes = 10

client_epochs_train =10
client_batch_size_train = 32
client_learning_rate_train = 0.001

client_epochs_fine_tune = 10
client_batch_size_fine_tune = 32
client_learning_rate_fine_tune = 0.001

client_batch_size_evaluate = 32

server_epochs_train = 10
server_batch_size_train = 64
server_learning_rate_train = 0.001

server_batch_size_evaluate = 64

def plot_average_loss(average_loss_df, filename='average_loss_plot.png'):
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plotting Average Test Loss
    plt.plot(average_loss_df['Iteration'], average_loss_df['Average Test Loss'], marker='o', color='b', label='Average Test Loss')

    # Plotting Average Train Loss
    plt.plot(average_loss_df['Iteration'], average_loss_df['Average Train Loss'], marker='x', color='r', label='Average Train Loss')

    # Adding title and labels
    plt.title(filename)  # Title of the graph
    plt.xlabel('Iteration')  # X-axis label
    plt.ylabel('Average Loss')  # Y-axis label

    # Add grid for better readability
    plt.grid()

    # Set x ticks for each iteration
    plt.xticks(range(len(average_loss_df['Iteration'])))

    # Add legend to distinguish the curves
    plt.legend()

    # Adjust layout to fit into the figure area
    plt.tight_layout()

    # Save the plot to the file
    plt.savefig(filename+".jpg")
    plt.close()  # Close the figure to free memory

def plot_loss_per_client(average_loss_df, filename='loss_plot.png',client_id = -1,server_split_ratio = 1):
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plotting Average Test Loss
    plt.plot(average_loss_df['Iteration'], average_loss_df['Test Loss'], marker='o', color='b', label='Average Test Loss')

    # Plotting Average Train Loss
    plt.plot(average_loss_df['Iteration'], average_loss_df['Train Loss'], marker='x', color='r', label='Average Train Loss')

    # Adding title and labels
    plt.title("Client "+str(client_id)+" Loss "+ "(Server data "+str(round(server_split_ratio,2))+")")  # Title of the graph

    plt.xlabel('Iteration')  # X-axis label
    plt.ylabel('Loss')  # Y-axis label

    # Add grid for better readability
    plt.grid()

    # Set x ticks for each iteration
    plt.xticks(range(len(average_loss_df['Iteration'])))

    # Add legend to distinguish the curves
    plt.legend()

    # Adjust layout to fit into the figure area
    plt.tight_layout()

    # Save the plot to the file
    plt.savefig(filename+".jpg")
    plt.close()  # Close the figure to free memory

