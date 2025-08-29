import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import epsilon
from main_ import *
from Graph_global import *
from config import AlgorithmSelected

# ---- shared style variables (fallbacks if not defined elsewhere) ----
try:
    axes_titles_font
except NameError:
    axes_titles_font = 16
try:
    axes_number_font
except NameError:
    axes_number_font = 14
try:
    tick_font_size
except NameError:
    tick_font_size = 14
try:
    legend_font_size
except NameError:
    legend_font_size = 16
try:
    linewidth
except NameError:
    linewidth = 3


def get_data_per_iter(rd):
    ans = {}
    for client_id, data_per_iter in rd.server_accuracy_per_client_1_max.items():
        for iter_, number_ in data_per_iter.items():
            if iter_ not in ans:
                ans[iter_] = []
            ans[iter_].append(number_)
    return ans


def get_data_for_graph_cluster(rds):
    """
    Build: {cluster_count -> list of accuracies at the best iteration (per rd)}
    """
    ans = {}
    for cluster, rd_list in rds.items():
        for rd in rd_list:
            data_per_iter = get_data_per_iter(rd)
            # choose iteration with highest average accuracy
            max_iter = max(data_per_iter, key=lambda k: np.mean(data_per_iter[k]))
            if cluster not in ans:
                ans[cluster] = []
            ans[cluster].extend(data_per_iter[max_iter])
    return ans


def create_dual_avg_plot(
    data_alpha1: dict,
    data_alpha5: dict,
    x_label="Amount of Clusters",
    y_label="Top-1 Accuracy (%)",
    output_path="figures/number_of_clusters_alpha_1_5.pdf",
    title=None
):
    """
    Plot two curves (α=1 and α=5) with mean ± SEM shading.
    Assumes input values are probabilities in [0,1]; converts to percent.
    """

    def _mean_sem_series(items_sorted):
        x = np.array([x for x, _ in items_sorted])
        means = []
        sems = []
        for _, vals in items_sorted:
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                means.append(np.nan)
                sems.append(0.0)
                continue
            m = float(np.mean(v))
            if v.size > 1:
                sd = float(np.std(v, ddof=1))
                se = sd / np.sqrt(v.size)
            else:
                se = 0.0
            means.append(m)
            sems.append(se)
        # RETURN RAW PROBABILITIES (no x100)
        return x, np.asarray(means), np.asarray(sems)

    # --- α = 1 ---
    items1 = sorted(data_alpha1.items())
    x1, m1, se1 = _mean_sem_series(items1)

    # --- α = 5 ---
    items5 = sorted(data_alpha5.items())
    x5, m5, se5 = _mean_sem_series(items5)

    # to %
    #m1, se1 = m1 * 100.0, se1 * 100.0
    #m5, se5 = m5 * 100.0, se5 * 100.0

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # α=1
    ax.plot(x1, m1, marker='o', linewidth=linewidth, markersize=4, label=r'$\alpha=1$')
    ax.fill_between(x1, m1 - se1, m1 + se1, alpha=0.2)

    # α=5
    ax.plot(x5, m5, marker='s', linewidth=linewidth, markersize=4, label=r'$\alpha=5$')
    ax.fill_between(x5, m5 - se5, m5 + se5, alpha=0.2)

    # labels & title — SAME SIZES as your other plots
    ax.set_xlabel(x_label, fontsize=axes_number_font)
    ax.set_ylabel(y_label, fontsize=axes_number_font)
    if title:
        ax.set_title(title, fontsize=axes_titles_font)

    ax.tick_params(axis='both', labelsize=tick_font_size)
    ax.grid(True, linestyle='--', alpha=0.35)

    # legend
    ax.legend(loc='upper left', fontsize=legend_font_size, frameon=True)

    plt.tight_layout()

    # save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    cluster_names = {"Optimal": "CBG", 1: "No Clusters"}  # Cluster By Group

    all_data = read_all_pkls("diff_clusters")
    merged_dict1 = merge_dicts(all_data)

    # Navigate to your slice: [...][25][5][alpha][1]
    root = merged_dict1[DataSet.CIFAR100.name][25][5]

    # Build rds for α=1 and α=5
    merged_alpha1 = root[1][1]
    merged_alpha5 = root[1][5]

    rds_alpha1 = switch_algo_and_seed_cluster(merged_alpha1, dich=5, data_type=DataSet.CIFAR100.name)
    rds_alpha5 = switch_algo_and_seed_cluster(merged_alpha5, dich=5, data_type=DataSet.CIFAR100.name)

    data_alpha1 = get_data_for_graph_cluster(rds_alpha1)
    data_alpha5 = get_data_for_graph_cluster(rds_alpha5)

    create_dual_avg_plot(
        data_alpha1,
        data_alpha5,
        x_label="Amount of Clusters",
        y_label="Top-1 Accuracy (%)",
        output_path="figures/number_of_clusters_alpha_1_5.pdf",
        title="Top-1 Accuracy at Best Iteration vs. Clusters"
    )
