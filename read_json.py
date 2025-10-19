#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_algorithms_avg_accuracy.py
Faceted by client_net_type_name + MAPL_C/S styling & shared colors by X.

Filters (Fig 1 only):
- ONLY includes JSONs where summary["data_set_selected"]["_name_"] == "CIFAR100" (case-insensitive safeguard)

New Figure (Fig 2):
- Only client_net_type_name == "rndWeak"
- Faceted by summary["data_set_selected"]["_name_"] (independent y-axes)
- Filter alpha_dich == --alpha (default 5)

Figure(s)
---------
1) algorithms_avg_accuracy_by_client_net_type.pdf
   - Facets (subplots) by client_net_type_name (from summary.client_net_type_name)
   - x-axis: iteration
   - y-axis: mean accuracy across ALL clients & ALL seeds
   - Lines: per displayed label
       * Non-MAPL: algorithm name (solid)
       * MAPL:
           - MAPL_C-<X>  (client_accuracy_per_client_1)  -> dashed
           - MAPL_S-<X>  (server_accuracy_per_client_1_max) -> solid
         where <X> = summary.server_net_type._value_

2) algorithms_avg_accuracy_rndWeak_by_dataset_alpha{ALPHA}.pdf
   - Filter: client_net_type_name == "rndWeak", alpha_dich == {ALPHA}
   - Facets by dataset (summary.data_set_selected._name_)
   - x-axis: iteration
   - y-axis: mean accuracy across ALL clients & ALL seeds (independent per subplot)
   - Same line styling as above

Usage
-----
  python make_fig_algorithms_avg_accuracy.py --root results --inspect
  python make_fig_algorithms_avg_accuracy.py --root results --figdir figures
  # With a different alpha for Fig 2
  python make_fig_algorithms_avg_accuracy.py --root results --alpha 3
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

TITLE_MAP = {
    "rndNet":    "{Alex, Res, Mobile, Squeeze}",
    "rndStrong": "{Alex, Res}",
    "rndWeak":   "{Mobile, Squeeze}",
}

# ---------- Recursive traversal ----------
def walk(obj: Json, path: Tuple[str, ...] = ()):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (f"[{i}]",))

def path_str(path: Optional[Tuple[str, ...]]):  # -> str
    return ".".join(path) if path else "N/A"

# ---------- Field finders ----------
def find_algorithm_name(root: Json):
    candidates: List[Tuple[Tuple[str, ...], str]] = []
    for p, v in walk(root):
        if isinstance(v, str) and v.strip():
            if p and p[-1].lower() in {"_name_", "name"} and any("algorithm" in seg.lower() for seg in p):
                candidates.append((p, v.strip()))
            if p and p[-1].lower() == "algorithm":
                candidates.append((p, v.strip()))
    if candidates:
        def score(path: Tuple[str, ...]) -> int:
            s = 0
            if path and path[-1].lower() in {"_name_", "name"}:
                s += 2
            if any("algorithm" in seg.lower() for seg in path):
                s += 2
            if path and path[-1].lower() == "algorithm":
                s += 3
            return s
        best = max(candidates, key=lambda t: score(t[0]))
        return best[1], best[0]
    return None, None

def find_seed_num(root: Json):
    for p, v in walk(root):
        if not p:
            continue
        key = p[-1].lower()
        if key in {"seed_num", "seed"}:
            try:
                return int(v), p
            except Exception:
                try:
                    return int(float(v)), p
                except Exception:
                    continue
    return None, None

def looks_like_client_accuracy_map(v: Any) -> bool:
    if not isinstance(v, dict) or not v:
        return False
    n_checked = 0
    nested_dict_seen = 0
    for _, inner in v.items():
        n_checked += 1
        if isinstance(inner, dict) and inner:
            for it_val in inner.values():
                if isinstance(it_val, (int, float, str)):
                    nested_dict_seen += 1
                    break
        if n_checked >= 5:
            break
    return nested_dict_seen > 0

def find_exact_key_map(root: Json, keyname: str):
    hits: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if p and p[-1] == keyname and isinstance(v, dict):
            hits.append((p, v))
    if hits:
        hits.sort(key=lambda t: len(t[0]))
        return hits[0][1], hits[0][0]
    return None, None

def find_client_accuracy_map_anywhere(root: Json):
    v, p = find_exact_key_map(root, "client_accuracy_per_client_1")
    if v is not None:
        return v, p
    candidates: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if looks_like_client_accuracy_map(v):
            candidates.append((p, v))
    if candidates:
        def cscore(pv: Tuple[Tuple[str, ...], Any]) -> int:
            p, _ = pv
            s = 0
            if any("summary" in seg.lower() for seg in p):
                s += 2
            if any("client" in seg.lower() for seg in p):
                s += 1
            return s
        candidates.sort(key=cscore, reverse=True)
        return candidates[0][1], candidates[0][0]
    return None, None

def find_server_accuracy_map(root: Json):
    v, p = find_exact_key_map(root, "server_accuracy_per_client_1_max")
    if v is not None:
        return v, p
    candidates: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if looks_like_client_accuracy_map(v) and any("server" in seg.lower() for seg in p):
            candidates.append((p, v))
    if candidates:
        def cscore(pv: Tuple[Tuple[str, ...], Any]) -> int:
            p, _ = pv
            s = 0
            if any("summary" in seg.lower() for seg in p):
                s += 2
            if any("server" in seg.lower() for seg in p):
                s += 1
            return s
        candidates.sort(key=cscore, reverse=True)
        return candidates[0][1], candidates[0][0]
    return None, None

def find_server_net_type_value(root: Json):
    for p, v in walk(root):
        if not p:
            continue
        if p[-1] in {"_value_", "value"} and isinstance(v, (str, int, float)):
            if any(seg.lower() == "server_net_type" for seg in p) and any(seg.lower() == "summary" for seg in p):
                return str(v), p
    best: Optional[Tuple[Tuple[str, ...], str]] = None
    best_score = -1
    for p, v in walk(root):
        if not p:
            continue
        if p[-1] in {"_value_", "value"} and isinstance(v, (str, int, float)):
            pl = [seg.lower() for seg in p]
            if any("server" in seg for seg in pl) and any("net" in seg for seg in pl):
                score = 1 + (2 if any("summary" in seg for seg in pl) else 0)
                if score > best_score:
                    best = (p, str(v))
                    best_score = score
    if best:
        return best[1], best[0]
    return None, None

def find_client_net_type_name(root: Json):
    for p, v in walk(root):
        if p and p[-1] == "client_net_type_name":
            return (str(v) if v is not None else None), p
    return None, None

def find_dataset_selected_name(root: Json):
    # Exact preferred
    cur = root
    try:
        if isinstance(cur, dict) and "summary" in cur:
            cur = cur["summary"]
            if isinstance(cur, dict) and "data_set_selected" in cur:
                cur2 = cur["data_set_selected"]
                if isinstance(cur2, dict) and "_name_" in cur2:
                    v = cur2["_name_"]
                    return (str(v) if v is not None else None), ("summary", "data_set_selected", "_name_")
    except Exception:
        pass
    # Heuristic fallback
    for p, v in walk(root):
        if not p or not isinstance(v, (str, int, float)):
            continue
        pl = [seg.lower() for seg in p]
        if any("summary" in seg for seg in pl) and any("data_set_selected" in seg for seg in pl):
            if p[-1].lower() in {"_name_", "name"}:
                return str(v), p
    return None, None

def find_alpha_dich(root: Json):
    for p, v in walk(root):
        if p and p[-1] == "alpha_dich":
            try:
                return int(v), p
            except Exception:
                try:
                    return int(float(v)), p
                except Exception:
                    return None, p
    return None, None

# ---------- Loading ----------
def load_rows(root: Path, *, dataset_filter: Optional[str] = None, inspect: bool = False) -> pd.DataFrame:
    """
    dataset_filter:
        - None -> include all datasets
        - "CIFAR100" -> include only CIFAR100 (case-insensitive)
    """
    rows: List[Dict[str, Any]] = []
    files_scanned = 0
    files_used = 0

    for p in root.glob("**/*.json"):
        if not p.is_file():
            continue

        files_scanned += 1
        try:
            with p.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}")
            continue

        ds_name, ds_path = find_dataset_selected_name(raw)
        if dataset_filter is not None:
            if ds_name is None or str(ds_name).strip().lower() != dataset_filter.strip().lower():
                if inspect:
                    print(f"[SKIP] {p} — dataset={ds_name!r} @ {path_str(ds_path)} (want '{dataset_filter}')")
                continue

        alg_name, alg_path = find_algorithm_name(raw)
        seed_num, seed_path = find_seed_num(raw)
        client_map, client_path = find_client_accuracy_map_anywhere(raw)
        server_map, server_path = find_server_accuracy_map(raw)
        server_net_val, server_net_path = find_server_net_type_value(raw)
        client_net_name, client_net_path = find_client_net_type_name(raw)
        alpha_dich, alpha_path = find_alpha_dich(raw)

        if inspect:
            print(f"\n--- Inspect: {p}")
            print(f"  dataset_selected._name_: {ds_name} @ {path_str(ds_path)}")
            print(f"  algorithm: {alg_name} @ {path_str(alg_path)}")
            print(f"  seed_num : {seed_num} @ {path_str(seed_path)}")
            print(f"  client_accuracy_per_client_1: @ {path_str(client_path)}")
            print(f"  server_accuracy_per_client_1_max: @ {path_str(server_path)}")
            print(f"  server_net_type._value_: {server_net_val} @ {path_str(server_net_path)}")
            print(f"  client_net_type_name: {client_net_name} @ {path_str(client_net_path)}")
            print(f"  alpha_dich: {alpha_dich} @ {path_str(alpha_path)}")

        is_mapl = bool(alg_name and "mapl" in alg_name.lower())
        used_any = False

        # 1) CLIENT measure: all algorithms (for MAPL: label as MAPL_C-X)
        if client_map is not None:
            display_label = (alg_name or "unknown_alg")
            if is_mapl:
                x = server_net_val if server_net_val is not None else "NA"
                display_label = f"MAPL_C-{x}"

            for client_id, iter_dict in client_map.items():
                if not isinstance(iter_dict, dict):
                    continue
                for iter_k, acc_v in iter_dict.items():
                    try:
                        iteration = int(iter_k)
                    except Exception:
                        try:
                            iteration = int(float(iter_k))
                        except Exception:
                            continue
                    try:
                        acc = float(acc_v)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "_path": str(p),
                            "dataset": ds_name or "unknown_dataset",
                            "algorithm": alg_name or "unknown_alg",
                            "algorithm_display": display_label,
                            "measure": "client",
                            "seed": seed_num,
                            "client_id": str(client_id),
                            "iteration": iteration,
                            "accuracy": acc,
                            "client_net_type_name": client_net_name or "unknown_client_net",
                            "alpha_dich": alpha_dich,
                        }
                    )
                    used_any = True

        # 2) SERVER measure: only for MAPL (label as MAPL_S-X)
        if is_mapl and server_map is not None:
            x = server_net_val if server_net_val is not None else "NA"
            display_label = f"MAPL_S-{x}"

            for client_id, iter_dict in server_map.items():
                if not isinstance(iter_dict, dict):
                    continue
                for iter_k, acc_v in iter_dict.items():
                    try:
                        iteration = int(iter_k)
                    except Exception:
                        try:
                            iteration = int(float(iter_k))
                        except Exception:
                            continue
                    try:
                        acc = float(acc_v)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "_path": str(p),
                            "dataset": ds_name or "unknown_dataset",
                            "algorithm": alg_name or "unknown_alg",
                            "algorithm_display": display_label,
                            "measure": "server",
                            "seed": seed_num,
                            "client_id": str(client_id),
                            "iteration": iteration,
                            "accuracy": acc,
                            "client_net_type_name": client_net_name or "unknown_client_net",
                            "alpha_dich": alpha_dich,
                        }
                    )
                    used_any = True

        if used_any:
            files_used += 1

    if inspect:
        want = dataset_filter if dataset_filter else "ALL"
        print(f"\nScanned {files_scanned} JSON files; usable for plot (dataset={want}): {files_used}")

    if not rows:
        return pd.DataFrame(columns=[
            "dataset", "algorithm_display", "measure", "seed", "client_id",
            "iteration", "accuracy", "client_net_type_name", "alpha_dich"
        ])
    return pd.DataFrame(rows)

# ---------- Color/linestyle utilities ----------
_mapl_label_re = re.compile(r"^MAPL_(?P<variant>[CS])-(?P<x>.+)$")

def _extract_mapl_x(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label)
    return m.group("x") if m else None

def _extract_mapl_variant(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label)
    return m.group("variant") if m else None

def _build_color_map(labels: List[str]):
    """Assign colors so that MAPL_* with the same X share a color; others get distinct colors."""
    cmap = plt.get_cmap("tab10")
    mapl_x_values = []
    non_mapl_labels = []
    for lab in labels:
        x = _extract_mapl_x(lab)
        if x is not None:
            mapl_x_values.append(x)
        else:
            non_mapl_labels.append(lab)
    unique_x = list(dict.fromkeys(mapl_x_values))  # preserve order

    color_by_x: Dict[str, Any] = {x: cmap(i % 10) for i, x in enumerate(unique_x)}
    color_by_label: Dict[str, Any] = {}

    # First map all MAPL labels by their X
    for lab in labels:
        x = _extract_mapl_x(lab)
        if x is not None:
            color_by_label[lab] = color_by_x[x]

    # Then assign remaining colors to non-MAPL labels
    next_color_idx = len(unique_x)
    for lab in labels:
        if lab not in color_by_label:
            color_by_label[lab] = cmap(next_color_idx % 10)
            next_color_idx += 1

    return color_by_label

def _linestyle_for(label: str) -> str:
    v = _extract_mapl_variant(label)
    if v == "S":
        return "-"   # solid
    if v == "C":
        return "--"  # dashed
    return "-"       # default solid for non-MAPL

# ---------- Plotting (Figure 1: faceted by client_net_type_name, CIFAR100 only) ----------
def plot_faceted_by_client_net(df: pd.DataFrame, outdir: Path):
    if df.empty:
        print("[WARN] No usable rows to plot.")
        return
    if "client_net_type_name" not in df.columns:
        print("[WARN] No 'client_net_type_name' field found; skipping faceted figure.")
        return

    # Precompute per-label colors globally so they stay consistent across facets
    all_labels = sorted(df["algorithm_display"].unique())
    color_by_label = _build_color_map(all_labels)

    # Aggregate: mean across clients & seeds per (label, iteration, client_net_type_name)
    grp = (
        df.groupby(["client_net_type_name", "algorithm_display", "iteration"], dropna=False)["accuracy"]
          .mean()
          .reset_index()
          .rename(columns={"accuracy": "avg_accuracy"})
    )

    nets = sorted(grp["client_net_type_name"].unique(), key=lambda x: (str(x).lower() if x is not None else ""))
    n = len(nets)
    if n == 0:
        print("[WARN] No unique client_net_type_name values found.")
        return

    # Compute global x/y limits for consistent scales across facets
    global_xmin = int(grp["iteration"].min())
    global_xmax = int(grp["iteration"].max())
    global_ymin = float(grp["avg_accuracy"].min())
    global_ymax = float(grp["avg_accuracy"].max())

    # One row, n columns; share both axes
    ncols = n
    nrows = 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5.0 * ncols, 4.0),
        sharex=True, sharey=True, squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, net in zip(axes_flat, nets):
        sub = grp[grp["client_net_type_name"] == net]
        labels_here = sorted(sub["algorithm_display"].unique())

        for lab in labels_here:
            sub_lab = sub[sub["algorithm_display"] == lab].sort_values("iteration")
            ax.plot(
                sub_lab["iteration"],
                sub_lab["avg_accuracy"],
                marker=None,           # curves only
                linewidth=2.0,
                label=str(lab),
                color=color_by_label.get(lab, None),
                linestyle=_linestyle_for(lab),
            )

        ax.set_title(TITLE_MAP.get(str(net), str(net)))
        ax.grid(False)
        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(global_ymin, global_ymax)

    # Hide any unused axes
    for ax in axes_flat[len(nets):]:
        ax.axis("off")

    # Shared axis labels
    try:
        fig.supxlabel("Iteration")
        fig.supylabel("Average Accuracy")
    except Exception:
        axes_flat[0].set_ylabel("Average Accuracy")
        axes_flat[-1].set_xlabel("Iteration")

    # De-duplicated legend
    handles, labels_leg = [], []
    for a in axes_flat[:n]:
        h, l = a.get_legend_handles_labels()
        handles += h; labels_leg += l
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels_leg) if not (l in seen or seen.add(l))]

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.80])  # reserve top
    if uniq:
        fig.legend(
            [h for h, _ in uniq], [l for _, l in uniq],
            loc="upper center", ncol=min(4, len(uniq)), frameon=False, bbox_to_anchor=(0.5, 0.92)
        )

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "algorithms_avg_accuracy_by_client_net_type.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {outfile}")

# ---------- Plotting (Figure 2: rndWeak only, faceted by dataset, independent y-axes, alpha filter) ----------
def plot_rndweak_by_dataset_alpha(df: pd.DataFrame, outdir: Path, alpha_value: int):
    if df.empty:
        print("[WARN] No usable rows to plot for rndWeak-by-dataset.")
        return

    need_cols = {"dataset", "client_net_type_name", "alpha_dich"}
    if not need_cols.issubset(df.columns):
        missing = need_cols - set(df.columns)
        print(f"[WARN] Missing columns for rndWeak figure: {missing}; skipping.")
        return

    # Filter
    sub = df[(df["client_net_type_name"].astype(str) == "rndWeak") & (df["alpha_dich"] == alpha_value)]
    if sub.empty:
        print(f"[WARN] No rows where client_net_type_name='rndWeak' and alpha_dich=={alpha_value}.")
        return

    # Aggregate: mean across clients & seeds per (dataset, label, iteration)
    grp = (
        sub.groupby(["dataset", "algorithm_display", "iteration"], dropna=False)["accuracy"]
           .mean()
           .reset_index()
           .rename(columns={"accuracy": "avg_accuracy"})
    )

    datasets = sorted(grp["dataset"].unique(), key=lambda x: (str(x).lower() if x is not None else ""))
    n = len(datasets)
    if n == 0:
        print("[WARN] No datasets found for rndWeak figure.")
        return

    # Colors: consistent across datasets
    all_labels = sorted(grp["algorithm_display"].unique())
    color_by_label = _build_color_map(all_labels)

    # Grid: choose 2 columns for nicer layout
    ncols = 2 if n >= 2 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.0 * ncols, 4.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, datasets):
        gds = grp[grp["dataset"] == ds]
        labels_here = sorted(gds["algorithm_display"].unique())
        for lab in labels_here:
            sub_lab = gds[gds["algorithm_display"] == lab].sort_values("iteration")
            ax.plot(
                sub_lab["iteration"],
                sub_lab["avg_accuracy"],
                marker=None,
                linewidth=2.0,
                label=str(lab),
                color=color_by_label.get(lab, None),
                linestyle=_linestyle_for(lab),
            )
        ax.set_title(str(ds))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)

    # Hide any unused axes
    for ax in axes_flat[len(datasets):]:
        ax.axis("off")

    # Single de-duplicated legend
    handles, labels_leg = [], []
    for a in axes_flat[:n]:
        h, l = a.get_legend_handles_labels()
        handles += h; labels_leg += l
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels_leg) if not (l in seen or seen.add(l))]

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    if uniq:
        fig.legend(
            [h for h, _ in uniq], [l for _, l in uniq],
            loc="upper center", ncol=min(4, len(uniq)), frameon=False, bbox_to_anchor=(0.5, 0.98)
        )

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"algorithms_avg_accuracy_rndWeak_by_dataset_alpha{alpha_value}.pdf"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {outfile}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"), help="Root results folder")
    ap.add_argument("--figdir", type=Path, default=Path("figures"), help="Where to save the PDFs")
    ap.add_argument("--inspect", action="store_true", help="Print per-file detected paths/keys")
    ap.add_argument("--alpha", type=int, default=5, help="alpha_dich value to filter for rndWeak-by-dataset figure")
    args = ap.parse_args()

    # Figure 1 (CIFAR100 only)
    df_cifar100 = load_rows(args.root, dataset_filter="CIFAR100", inspect=args.inspect)
    print(f"[CIFAR100] Loaded rows: {len(df_cifar100):,}")
    if not args.inspect and not df_cifar100.empty:
        plot_faceted_by_client_net(df_cifar100, args.figdir)

    # Figure 2 (all datasets, then filtered inside the plot function)
    df_all = load_rows(args.root, dataset_filter=None, inspect=False if args.inspect else False)
    print(f"[ALL DATASETS] Loaded rows: {len(df_all):,}")
    if not args.inspect and not df_all.empty:
        plot_rndweak_by_dataset_alpha(df_all, args.figdir, alpha_value=args.alpha)

if __name__ == "__main__":
    main()
