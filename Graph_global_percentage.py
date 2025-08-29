from Graph_global import *
from main_ import *
import os
import math
from collections import defaultdict
from typing import Dict, List, Union, Any, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Debug toggle
# -----------------------------
DEBUG = True
def dbg(*args):
    if DEBUG:
        print("[DBG]", *args)

# -----------------------------
# Styling variables
# -----------------------------
axes_titles_font = 16
axes_number_font = 14       # axis label font size
legend_font_size = 16
tick_font_size = 14
linewidth = 3


# Fixed order for global-percentage curves (as percents)
GL_ORDER = [10, 25, 50, 75, 100]
COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
COLOR_MAP = {p: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, p in enumerate(GL_ORDER)}

# -----------------------------
# Normalizers
# -----------------------------
def norm_alpha(x) -> Union[int, None]:
    """Coerce alpha key to int if numeric ('1' -> 1, 1.0 -> 1); else None."""
    try:
        return int(round(float(x)))
    except Exception:
        return None

def norm_gl_pre(x) -> Union[int, None]:
    """
    Normalize global-percentage key to integer percent in {10,25,50,75,100}.
    Accepts '75%', 0.75, 75.0, 75, etc.
    Returns None if not parseable.
    """
    if isinstance(x, str):
        s = x.strip().rstrip('%')
    else:
        s = x
    try:
        val = float(s)
    except Exception:
        return None
    if 0.0 < val <= 1.0:  # treat 0..1 as ratio
        val *= 100.0
    return int(round(val))

def fmt_pct_label(pct_int: int) -> str:
    return f"{int(pct_int)}%"

# -----------------------------
# Aggregation
# -----------------------------

# -----------------------------
# Plotting
# -----------------------------
def plot_accuracy_curves_two_alphas(
    data_by_alpha: Dict[int, Dict[int, List[Run]]],  # alpha -> gl_pre(int %) -> runs
    alphas: Iterable[int] = (1, 5),
    y_label: str = "Top-1 Accuracy (%)",
    x_label: str = "Iteration",
    shaded_band: str = "sem",   # "sem" (default) | "std" | "ci95"
    add_start_point: bool = True,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=False)

    dbg("available gl% per alpha:", {a: sorted(list(data_by_alpha.get(a, {}).keys())) for a in alphas})

    # We’ll build legend entries only for series that actually appear
    legend_entries = []  # (label, color)

    for ax, alpha in zip(axes, alphas):
        gl_map = data_by_alpha.get(alpha, {})

        # Plot in fixed GL_ORDER so colors are stable
        for gl_pre in GL_ORDER:
            runs = gl_map.get(gl_pre)
            if not runs:
                continue

            iterations, means, stds, counts = aggregate_all_clients_seeds(runs, add_start_point)
            if len(iterations) == 0:
                continue

            if shaded_band == "sem":
                band = np.where(counts > 0, stds / np.sqrt(np.maximum(counts, 1)), 0.0)
            elif shaded_band == "ci95":
                sem  = np.where(counts > 0, stds / np.sqrt(np.maximum(counts, 1)), 0.0)
                band = 1.96 * sem
            else:  # "std"
                band = stds

            color = COLOR_MAP.get(gl_pre, None)
            label = fmt_pct_label(gl_pre)

            iters_np = np.array(iterations, dtype=int)
            ax.plot(iters_np, means, label=label, linewidth=linewidth, color=color)
            ax.fill_between(iters_np, means - band, means + band, alpha=0.20, color=color)

            # Track for legend once per unique gl_pre that appears anywhere
            if (label, color) not in legend_entries:
                legend_entries.append((label, color))

        ax.set_title(rf'$\alpha = {alpha}$', fontsize=axes_titles_font)
        ax.set_xlabel(x_label, fontsize=axes_number_font)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(axis='both', labelsize=tick_font_size)

    axes[0].set_ylabel(y_label, fontsize=axes_number_font)

    # Shared legend: only labels/colors that actually appeared
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

    # Save as PDF
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "server_data_alpha_1_5.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    dbg("saved figure to:", out_path)

    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    cluster_names = {"Optimal": "CBG", 1: "No Clusters"}  # unused here but kept

    # Load and merge your data as before
    all_data = read_all_pkls("diff_global_percentage")
    merged_dict1 = merge_dicts(all_data)

    # Adjust these selectors as needed
    merged_dict = merged_dict1[DataSet.CIFAR100.name][25][5]  # {gl_pre: {alpha: {seed: dict3}}}

    # Build data_by_alpha = {1: {gl_pre(int %): [runs...]}, 5: {...}}
    data_by_alpha: Dict[int, Dict[int, List[Run]]] = {
        1: defaultdict(list),
        5: defaultdict(list),
    }

    for gl_pre_raw, dict_by_alpha in merged_dict.items():
        gl_pre = norm_gl_pre(gl_pre_raw)
        if gl_pre is None:
            dbg("skip gl_pre (unparseable):", repr(gl_pre_raw))
            continue

        for alpha_raw, seeds_dict in dict_by_alpha.items():
            a = norm_alpha(alpha_raw)
            if a not in (1, 5):
                continue
            if not isinstance(seeds_dict, dict):
                continue

            for seed_, dict3 in seeds_dict.items():
                try:
                    rd = (dict3[AlgorithmSelected.PseudoLabelsClusters.name]
                              [NetsType.C_alex_S_vgg.name]["multi_model"]["max"]
                              ["greedy_elimination_L2"]["similar_to_cluster"][0]
                              ["withWeights"]["withInputConsistency"])
                except (KeyError, IndexError, TypeError) as e:
                    dbg(f"skip seed={seed_} gl_pre={gl_pre} alpha={a} due to path:", repr(e))
                    continue

                # mapping: {client_id: {iteration: accuracy}}
                measure_per_iter = getattr(rd, "client_accuracy_per_client_1", None)
                if not isinstance(measure_per_iter, dict):
                    dbg(f"skip seed={seed_} gl_pre={gl_pre} alpha={a}: unexpected run type {type(measure_per_iter).__name__}")
                    continue

                data_by_alpha[a][gl_pre].append(measure_per_iter)

    dbg("built data_by_alpha keys:",
        {a: sorted(list(inner.keys())) for a, inner in data_by_alpha.items()})

    # Plot both alphas side-by-side with shared legend on top
    plot_accuracy_curves_two_alphas(
        data_by_alpha=data_by_alpha,
        alphas=(1, 5),
        y_label="Top-1 Accuracy (%)",
        x_label="Iteration",
        shaded_band="sem",
        add_start_point=True
    )
