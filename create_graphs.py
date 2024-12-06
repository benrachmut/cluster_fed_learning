# File path to your pickle file
import pickle
from config import *
file_path = 'num_clusters_1_Mix_Percentage_0.2_Epochs_2_Iterations_20_Server_Split_Ratio_0.2_Num_Classes_2_Identical_Clients_1.pkl'

# Open and read the pickle file
with open(file_path, 'rb') as file:
    data_ = pickle.load(file)



