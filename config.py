from enum import Enum

from matplotlib import pyplot as plt


class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

seed_num= 1
#is_soft_loss = True
with_server_net = False
with_prev_weights = False
epochs_num_input =2
iterations = 10
num_clients = 2
percent_train_data_use = 0.01
percent_test_relative_to_train = 0.1
server_split_ratio = 0.2
server_net_type = NetType.VGG
client_net_type = NetType.ALEXNET
num_classes = 10
client_batch_size_train = 32
client_learning_rate_train = 0.001
client_batch_size_fine_tune = 32
client_learning_rate_fine_tune = 0.001
client_batch_size_evaluate = 32
server_batch_size_train = 32
server_learning_rate_train = 0.001

server_batch_size_evaluate = 32




def get_meta_data():
    ans = {
        'c_amount':[num_clients],
        'seed':[seed_num],
        'server_data': [server_split_ratio],
        'is_server_net': [with_server_net],  # You might need to pass or save client_split_ratio
        'is_prev_weights': [with_prev_weights],
        'epochs': [epochs_num_input],
        'percent_train_data': [percent_train_data_use]
    }
    return ans

def get_meta_data_text_keys():
    ans = []
    for k in get_meta_data().keys():
        ans.append(k)
    return ans

def file_name():
    ans = ""
    for k,v in get_meta_data().items():
        ans = ans+k+"_"+str(v[0])+"__"
    return ans






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

