from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected


if __name__ == '__main__':


    all_data = read_all_pkls("fix_pkl")
    merged_dict1 = merge_dicts(all_data)




    with open('all_algos_dich_1.pkl', 'wb') as f:
        pickle.dump(merged_dict1, f)
    print()


