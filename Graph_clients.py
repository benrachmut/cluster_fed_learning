# -*- coding: utf-8 -*-
"""
Bar chart: Top-1 accuracy at the last iteration (over all seeds) vs. number of clients.
- Uses SEM error bars
- Alpha = 1 only (if 'alpha' attr exists and !=1, it's skipped)
- Y axis is raw probabilities in [0,1] (NOT percent)
- Saves PDF to figures/num_clients_last_iter_alpha1.pdf
"""

from Graph_global import  *
from main_ import  *
# ---- shared style variables (fallbacks if not already defined) ----
try: axes_titles_font
except NameError: axes_titles_font = 16
try: axes_number_font
except NameError: axes_number_font = 14
try: tick_font_size
except NameError: tick_font_size = 14
try: legend_font_size
except NameError: legend_font_size = 16
try: linewidth
except NameError: linewidth = 3

# ======== CONFIG: point to your file ========
PICKLE_PATH = r"diff_clients"
OUTPUT_PDF  = "figures/num_clients_last_iter_alpha1.pdf"
TITLE       = "Top-1 Accuracy at Last Iteration vs. Number of Clients (α=1)"
X_LABEL     = "Number of Clients"
Y_LABEL     = "Top-1 Accuracy"   # raw 0..1


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _iter_rd(obj):
    """Yield any objects that look like a 'run datum' with the needed field."""
    try:
        # custom class with the field
        getattr(obj, "server_accuracy_per_client_1_max")
        yield obj
        return
    except Exception:
        pass

    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_rd(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _iter_rd(v)
    # else: ignore scalars/others


def _last_iter_mean_over_clients(rd):
    """
    Given an rd with rd.server_accuracy_per_client_1_max : {client_id: {iter: acc}},
    compute mean accuracy at the LAST iteration across all clients that have that iter.
    Returns (n_clients_detected, mean_acc_at_last_iter) or (None, None) if not possible.
    """
    per_client = getattr(rd, "server_accuracy_per_client_1_max", None)
    if not isinstance(per_client, dict) or not per_client:
        return None, None

    # determine number of clients & last iteration across all clients
    n_clients = len(per_client)
    all_iters = []
    for _, it2acc in per_client.items():
        if isinstance(it2acc, dict) and it2acc:
            try:
                all_iters.append(max(int(k) for k in it2acc.keys()))
            except Exception:
                pass
    if not all_iters:
        return n_clients, None
    last_iter = max(all_iters)

    # collect accuracies for last_iter
    vals = []
    for _, it2acc in per_client.items():
        if isinstance(it2acc, dict) and last_iter in it2acc:
            try:
                v = float(it2acc[last_iter])
                if np.isfinite(v):
                    vals.append(v)
            except Exception:
                continue

    if not vals:
        return n_clients, None

    return n_clients, float(np.mean(vals))


def _sem(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size <= 1:
        return 0.0
    return float(np.std(a, ddof=1) / np.sqrt(a.size))


def main():
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    obj = read_all_pkls(PICKLE_PATH)

    # gather last-iter means grouped by number of clients (over all seeds/runs)
    by_nclients = {}  # n_clients -> list of last-iter means across seeds
    n_examined = 0
    n_used = 0
    for rd in _iter_rd(obj):
        n_examined += 1
        # If alpha attribute exists and is not 1, skip
        alpha = getattr(rd, "alpha", 1)
        if alpha != 1:
            continue

        n_clients, mean_at_last = _last_iter_mean_over_clients(rd)
        if (n_clients is None) or (mean_at_last is None):
            continue

        by_nclients.setdefault(n_clients, []).append(mean_at_last)
        n_used += 1

    if not by_nclients:
        print("No usable runs found (alpha=1) with last-iteration accuracies.")
        return

    # Prepare bar data
    xs = sorted(by_nclients.keys())
    means = np.array([np.mean(by_nclients[k]) for k in xs], dtype=float)
    sems  = np.array([_sem(by_nclients[k]) for k in xs], dtype=float)

    # Plot (single figure)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(xs)), means, yerr=sems, capsize=4, linewidth=linewidth)

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(k) for k in xs], fontsize=tick_font_size)
    ax.set_xlabel(X_LABEL, fontsize=axes_number_font)
    ax.set_ylabel(Y_LABEL, fontsize=axes_number_font)
    ax.set_title(TITLE, fontsize=axes_titles_font)

    ax.tick_params(axis='y', labelsize=tick_font_size)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)

    plt.tight_layout()
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    print(f"Saved: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
