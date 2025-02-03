import pickle
from enum import Enum
import matplotlib.pyplot as plt

from main_ import RecordData

def get_data(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)


    ans = {}
    for selected_cluster in selected_clusters:
        ans[selected_cluster] = data[selected_cluster]["C_alex_S_vgg"]["manual"]
        #for feed_back in feed_backs:
        temp_data =  data[selected_cluster]["C_alex_S_vgg"]["manual"][selected_feedback]
        if selected_measure == Measure.Average_Clients_Accuracy:
            ans[selected_cluster] =temp_data.client_accuracy_per_client_1
        if selected_measure == Measure.Average_Server_Accuracy:
            ans[selected_cluster] =temp_data.server_accuracy_per_client_1

    return ans


def get_average_per_cluster(data_):
    ans = {}
    to_mean = {}
    for cluster_num, dict_1 in data_.items():
        ans[cluster_num] = {}
        to_mean[cluster_num] = {}
        for client_id, dict_2 in dict_1.items():
            for i, measure in dict_2.items():
                if i not in to_mean[cluster_num]:
                    to_mean[cluster_num][i] = []
                to_mean[cluster_num][i].append(measure)
    ans = {}
    for cluster_num, dict_1 in to_mean.items():
        ans[cluster_num] = {}
        for i, measure_list in dict_1.items():
            ans[cluster_num][i] = sum(measure_list) / len(measure_list)
    return ans

class Measure (Enum):
    Average_Clients_Accuracy=1
    Average_Server_Accuracy=2
if __name__ == '__main__':
    selected_clusters = [1,3,5]
    selected_measure = Measure.Average_Clients_Accuracy
    selected_feedback = "similar_to_cluster"#["similar_to_client", "similar_to_cluster"]

    data_ = get_data("C_alex_S_vggNonIIDfull_4_manual_similar_to_client.pkl")
    data_average = get_average_per_cluster(data_)
    colors = ['b', 'c', 'm', 'y', 'k']  # Add more colors if needed

    plt.figure(figsize=(10, 6))

    # Iterate over each key-value pair in data_average
    for i, (color_key, values) in enumerate(data_average.items()):
        x_values = list(values.keys())  # X-axis
        y_values = list(values.values())  # Y-axis

        plt.plot(x_values, y_values, marker='o', linestyle='-', color=colors[i % len(colors)],
                 label=f'Clusters {color_key}')

    # Labels and title
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy %")
    plt.title(selected_measure.name+" for Personalized Validation Data")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()



