from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected


if __name__ == '__main__':


    all_data = read_all_pkls("fix_pkl")
    merged_dict1 = merge_dicts(all_data)
    del merged_dict1[DataSet.CIFAR10.name][25][5][0.2][5][1][AlgorithmSelected.FedAvg.name]
    del merged_dict1[DataSet.CIFAR10.name][25][5][0.2][5][2][AlgorithmSelected.FedAvg.name]
    del merged_dict1[DataSet.CIFAR10.name][25][5][0.2][5][3][AlgorithmSelected.FedAvg.name]

    with open('merged_dict1.pkl', 'wb') as f:
        pickle.dump(merged_dict1, f)
    print()


