# File path to your pickle file
import pickle
from main_ import *
from config import *
file_path = 'num_clusters_1_Mix_Percentage_0.2_Epochs_10_Iterations_10_Server_Split_Ratio_0.5_Num_Classes_2_Identical_Clients_2.pkl'

# Open and read the pickle file
with open(file_path, 'rb') as file:
    data_ = pickle.load(file)
print()


