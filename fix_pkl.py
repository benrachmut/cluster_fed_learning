from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected


if __name__ == '__main__':


    all_data = read_all_pkls("fix_pkl")
    merged_dict1 = merge_dicts(all_data)
    del merged_dict1[DataSet.EMNIST_balanced.name][25][5][0.2][100][AlgorithmSelected.PseudoLabelsClusters.name]

    with open('merged_dict1.pkl', 'wb') as f:
        pickle.dump(merged_dict1, f)
    print()


