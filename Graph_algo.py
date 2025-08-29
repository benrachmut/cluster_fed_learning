from Graph_global import *
from main_ import *
import os
import math
from typing import Dict, List, Union, Any, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Debug toggle

# -----------------------------
# Styling variables
# -----------------------------
axes_titles_font = 16
axes_number_font = 14       # axis label font size
legend_font_size = 16
tick_font_size = 14
linewidth = 3

import os
import numpy as np
import matplotlib.pyplot as plt

# Reuse your types & helpers
Run = Dict[Any, Dict[Union[int, str], float]]  # {client_id: {iteration: accuracy}}


from Graph_global import *
ALGO_COLOR_MAP = {}
canon_algo = lambda s: s  # identity fallback
from collections import defaultdict

# If you have a preferred order for algorithms, set it here.
# Otherwise we’ll infer from the data keys per alpha.
ALGO_ORDER = None  # e.g., ["FreeForm", "MainPseudocodeOnly", "CompletePseudoCode", "DSA", "Oracle"]

def plot_accuracy_curves_two_alphas_by_algo(
    data_by_alpha: Dict[int, Dict[str, List[Run]]],  # alpha -> algo_key -> runs
    alphas: Iterable[int] = (1, 5),
    y_label: str = "Top-1 Accuracy (%)",
    x_label: str = "Iteration",
    shaded_band: str = "sem",   # "sem" | "std" | "ci95"
    add_start_point: bool = True,
    outfile_tag: str = "by_algo"
):
    """
    Plot accuracy curves for two alphas, with one series per ALGORITHM
    (gl_pre is assumed constant across runs).
    data_by_alpha: {alpha: {algo_key: [Run, ...]}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=False)

    # Build legend entries only for series that actually appear
    legend_entries = []  # (label, color)

    for ax, alpha in zip(axes, alphas):
        algo_map = data_by_alpha.get(alpha, {})
        if not algo_map:
            ax.set_title(rf'$\alpha = {alpha}$', fontsize=axes_titles_font)
            ax.set_xlabel(x_label, fontsize=axes_number_font)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(axis='both', labelsize=tick_font_size)
            continue

        # Decide plotting order
        algo_iter = ALGO_ORDER if ALGO_ORDER else sorted(algo_map.keys())

        for algo_key in algo_iter:
            runs = algo_map.get(algo_key)
            if not runs:
                continue

            iterations, means, stds, counts = aggregate_all_clients_seeds(runs, add_start_point)
            if len(iterations) == 0:
                continue

            # Shaded uncertainty
            if shaded_band == "sem":
                band = np.where(counts > 0, stds / np.sqrt(np.maximum(counts, 1)), 0.0)
            elif shaded_band == "ci95":
                sem  = np.where(counts > 0, stds / np.sqrt(np.maximum(counts, 1)), 0.0)
                band = 1.96 * sem
            else:  # "std"
                band = stds

            color = ALGO_COLOR_MAP.get(algo_key, None)  # fall back to Matplotlib cycle if None
            label = canon_algo(algo_key)

            iters_np = np.array(iterations, dtype=int)
            ax.plot(iters_np, means, label=label, linewidth=linewidth, color=color)
            ax.fill_between(iters_np, means - band, means + band, alpha=0.20, color=color)

            # Track for legend once per unique algo that appears anywhere
            pair = (label, color)
            if pair not in legend_entries:
                legend_entries.append(pair)

        ax.set_title(rf'$\alpha = {alpha}$', fontsize=axes_titles_font)
        ax.set_xlabel(x_label, fontsize=axes_number_font)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis='both', labelsize=tick_font_size)

    axes[0].set_ylabel(y_label, fontsize=axes_number_font)

    # Shared legend
    if legend_entries:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=c, lw=2) for (lbl, c) in legend_entries]
        labels  = [lbl for (lbl, _c) in legend_entries]
        fig.legend(
            handles, labels,
            loc='upper center',
            ncol=max(1, len(labels)),
            fontsize=legend_font_size,
            frameon=True,
            bbox_to_anchor=(0.5, 1.02),
            borderaxespad=0.2
        )
        fig.subplots_adjust(top=0.90, wspace=0.10)

    # Save
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    a_list = list(alphas)
    out_path = os.path.join(out_dir, f"server_data_alpha_{a_list[0]}_{a_list[1]}_{outfile_tag}.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")

    plt.show()

if __name__ == '__main__':

    # Load and merge your data as before
    all_data = read_all_pkls("diff_algo")
    merged_dict1 = merge_dicts(all_data)

    # Adjust these selectors as needed
    merged_dict = merged_dict1[DataSet.CIFAR100.name][25][5]  # {gl_pre: {alpha: {seed: dict3}}}

    # Build data_by_alpha = {1: {gl_pre(int %): [runs...]}, 5: {...}}
    data_ ={}
    for alpha,dict1 in merged_dict.items():
        if alpha not in data_:
            data_[alpha]={}
        dict_2 = dict1[5]
        for seed_,dict_3 in dict_2.items():

            for algo,dict_4 in dict_3.items():
                if algo not in data_[alpha]:
                    data_[alpha][algo] = []
                if algo == "SCAFFOLD":
                    rd = dict_4["C_alex_S_alex"]["no_model"]["mean"]["kmeans"]["similar_to_client"][1]
                    data_[alpha][algo].append(rd.client_accuracy_per_client_1)
                else:
                    print()
    plot_accuracy_curves_two_alphas_by_algo(data_, alphas=(1,5))




