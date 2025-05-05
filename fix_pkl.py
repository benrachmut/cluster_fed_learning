from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected


if __name__ == '__main__':


    all_data = read_all_pkls("fix_pkl")
    merged_dict1 = merge_dicts(all_data)

    ans = {}
    for k1,v1 in merged_dict1.items():
        ans[k1] ={}
        for k2,v2 in v1.items():
            ans[k1][k2] = {}
            for k3,v3 in v2.items():
                ans[k1][k2][k3] = {}
                for k4, v4 in v3.items():
                    ans[k1][k2][k3][k4] = {}
                    for k5,v5 in v4.items():
                        ans[k1][k2][k3][k4][k5] = {}
                        for k6,v6 in v5.items():
                            if k6!="FedAvg":
                                ans[k1][k2][k3][k4][k5][1]={k6:v6}
    print()



    with open('all_algos_dich_1.pkl', 'wb') as f:
        pickle.dump(ans, f)
    print()


