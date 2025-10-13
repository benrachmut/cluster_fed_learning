from old_graphs_python_files.Graph_global import *

if __name__ == '__main__':

    all_data = read_all_pkls("diff_global_percentage")
    merged_dict1 = merge_dicts(all_data)

    merged_dict = merged_dict1[DataSet.CIFAR100.name][25][5]  # {gl_pre: dict_by_alpha}

    for gl_pre, dict_by_alpha in merged_dict.items():
        # Safely remove alpha==5 if it exists
        dict_by_alpha.pop(5, None)  # <- no iteration over dict_by_alpha
    print()



    with open('diff_glob_alpha1.pkl', 'wb') as f:
        pickle.dump(merged_dict1, f)
    print()


