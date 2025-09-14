from Graph_global import *
from main_ import *

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

# ------- font fallbacks if your constants aren't defined -------
try:
    _AX_NUM = axes_number_font
    _AX_TITLE = axes_titles_font
    _TICK = tick_font_size
    _LEG = legend_font_size
except NameError:
    _AX_NUM = 12
    _AX_TITLE = 12
    _TICK = 10
    _LEG = 10


def _roles_for_algo(algo_name: str):
    """For MAPL: plot Clients + Server, else Clients only."""
    return ["Clients", "Server"] if "mapl" in algo_name.lower() else ["Clients"]


def _linestyle_for(role: str):
    """Solid for Clients, dashed for Server."""
    return "-" if role == "Clients" else "--"


def _series_mean_ci(vals, confidence=0.95):
    """Mean ± normal CI; ignore NaNs."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.nanmean(arr))
    if arr.size <= 1:
        return mean, mean, mean
    stderr = stats.sem(arr, nan_policy="omit")
    h = stderr * stats.norm.ppf((1 + confidence) / 2.0)
    return mean, mean - h, mean + h


def plot_algos_by_dich_and_net(
    data_for_graph: dict,
    *,
    x_label="Iterations",
    y_label="Top-1 Accuracy (%)",
    confidence=0.95,
    figsize=(16, 6),
    savepath=None
):
    """
    Expects:
      data_for_graph[dich][algorithm][net_type][entity]["iteration"] -> list[float]
      where entity ∈ {"Clients","Server"}.

    Subplots:
      rows = sorted dich values
      cols = sorted net_type values

    Curves:
      - MAPL: Clients (solid) + Server (dashed)
      - Others: Clients only (solid)
    """
    if not data_for_graph:
        raise ValueError("data_for_graph is empty.")

    # Row order: dich
    dich_keys = list(data_for_graph.keys())
    try:
        dich_keys = sorted(dich_keys, key=lambda x: float(x))
    except Exception:
        dich_keys = sorted(dich_keys, key=str)

    # Column order: net types across all dich
    net_types = set()
    for d in data_for_graph:
        for algo in data_for_graph[d]:
            for net in data_for_graph[d][algo]:
                net_types.add(net)
    net_types = sorted(net_types, key=str)

    # Algorithms (for colors)
    algos = set()
    for d in data_for_graph:
        for algo in data_for_graph[d]:
            algos.add(algo)
    algo_list = sorted(algos, key=str)

    # Colors per algorithm
    tab10 = plt.get_cmap("tab10").colors
    color_cycle = itertools.cycle(tab10)
    algo_colors = {algo: next(color_cycle) for algo in algo_list}

    # Pre-scan for global y-lims from actually plotted series
    all_vals = []
    for d in dich_keys:
        for net in net_types:
            for algo in algo_list:
                for role in _roles_for_algo(algo):
                    role_dict = (
                        data_for_graph.get(d, {})
                                       .get(algo, {})
                                       .get(net, {})
                                       .get(role, {})
                    )
                    if not isinstance(role_dict, dict):
                        continue
                    for _, vals in role_dict.items():
                        if isinstance(vals, (list, tuple, np.ndarray)):
                            for v in vals:
                                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                    all_vals.append(v)

    if not all_vals:
        raise ValueError("No numeric values found to plot.")

    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    pad = max(1e-6, 0.03 * (vmax - vmin) if vmax > vmin else 1.0)
    ymin, ymax = float(vmin - pad), float(vmax + pad)

    # Figure
    nrows, ncols = len(dich_keys), len(net_types)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Legend handles (one per algorithm color)
    algo_handles = {}
    # Role style legend (Clients/Server key)
    role_proxies = [
        Line2D([0], [0], linestyle="-", linewidth=2.0, color="black", label="Clients"),
        Line2D([0], [0], linestyle="--", linewidth=2.0, color="black", label="Server"),
    ]

    for r, d in enumerate(dich_keys):
        for c, net in enumerate(net_types):
            ax = axs[r, c]

            for algo in algo_list:
                roles = _roles_for_algo(algo)
                for role in roles:
                    role_dict = (
                        data_for_graph.get(d, {})
                                       .get(algo, {})
                                       .get(net, {})
                                       .get(role, {})
                    )
                    if not isinstance(role_dict, dict) or not role_dict:
                        continue

                    # Iterations as sorted numeric if possible
                    try:
                        iters = sorted(role_dict.keys(), key=lambda x: float(x))
                    except Exception:
                        iters = sorted(role_dict.keys(), key=str)

                    xs, means, lbs, ubs = [], [], [], []
                    for it in iters:
                        m, lb, ub = _series_mean_ci(role_dict[it], confidence=confidence)
                        xs.append(float(it))
                        means.append(m)
                        lbs.append(lb)
                        ubs.append(ub)

                    xs = np.asarray(xs, dtype=float)
                    means = np.asarray(means, dtype=float)
                    lbs = np.asarray(lbs, dtype=float)
                    ubs = np.asarray(ubs, dtype=float)
                    mask = ~np.isnan(means)
                    if not np.any(mask):
                        continue
                    xs, means, lbs, ubs = xs[mask], means[mask], lbs[mask], ubs[mask]

                    color = algo_colors[algo]
                    ls = _linestyle_for(role)
                    label = f"{algo} (Server)" if role == "Server" else algo

                    line, = ax.plot(xs, means, linestyle=ls, linewidth=2.0, color=color, label=label)
                    ax.fill_between(xs, lbs, ubs, alpha=0.2, color=color)

                    # Capture one handle per algorithm (Clients) for the color legend
                    if role == "Clients" and algo not in algo_handles:
                        algo_handles[algo] = line

            ax.set_title(f"dich={d} | net={net}", fontsize=_AX_TITLE)
            ax.set_xlabel(x_label, fontsize=_AX_TITLE)
            if c == 0:
                ax.set_ylabel(y_label, fontsize=_AX_TITLE)
            ax.tick_params(axis="both", labelsize=_TICK)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.15, linestyle=":")

    # Figure-level legends
    if algo_handles:
        fig.legend(
            list(algo_handles.values()),
            list(algo_handles.keys()),
            loc="upper center",
            ncol=max(1, min(len(algo_handles), 5)),
            frameon=False,
            fontsize=_LEG,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.legend(
        role_proxies,
        [h.get_label() for h in role_proxies],
        loc="upper right",
        frameon=False,
        fontsize=_LEG,
        bbox_to_anchor=(0.98, 0.98),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, axs


if __name__ == "__main__":
    # --- load & reshape like your current script ---
    all_data = read_all_pkls("diff_algo")
    merged_dict1 = merge_dicts(all_data)

    data_type = DataSet.CIFAR100.name
    top_what = 1
    data_for_graph = {}

    for dich in [5]:
        merged_dict = merged_dict1[data_type][25][5][1][dich]
        merged_dict = switch_algo_and_seedV3(merged_dict)
        # Collect into: data_for_graph[dich][algorithm][net_type][entity]["iteration"] -> list[float]
        data_for_graph[dich] = collect_data_per_server_client_iterationV2(
            merged_dict, top_what, data_type
        )

    # --- plot ---
    fig, axs = plot_algos_by_dich_and_net(
        data_for_graph,
        x_label="Iterations",
        y_label="Top-1 Accuracy (%)",
        confidence=0.95,
        figsize=(16, 6),
        savepath="figures/client_server_alpha.pdf",
    )
    plt.show()
