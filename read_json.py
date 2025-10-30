#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_algorithms_avg_accuracy.py

Stable colors across ALL figures:
- Every figure now uses the same global <algo -> color> map.
- MAPL variants (MAPL_C-*, MAPL_S-*) share the base "MAPL" color,
  but differ by linestyle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# ---------------------------------------------------------------------
# 1) TITLES
# ---------------------------------------------------------------------
TITLE_MAP = {
    "rndNet":    "{AlexNet, ResNet, MobileNet, SqueezeNet}",
    "rndStrong": "{AlexNet, ResNet}",
    "rndWeak":   "{MobileNet, SqueezeNet}",
}

TITLE_NET_MAP = {
    "ALEXNET": "AlexNet",
    "MobileNet": "MobileNet",
}

def _net_title_from_map(*candidates: str) -> str:
    for c in candidates:
        if c is None:
            continue
        k = str(c).strip()
        if not k:
            continue
        if k in TITLE_NET_MAP:
            return TITLE_NET_MAP[k]
        kl = k.lower()
        for ref in TITLE_NET_MAP.keys():
            if ref.lower() == kl:
                return TITLE_NET_MAP[ref]
    for c in candidates:
        if c and str(c).strip():
            return str(c).strip()
    return "clients"

# ---------------------------------------------------------------------
# 2) GLOBAL STABLE COLOR MAP
# ---------------------------------------------------------------------
# This is the ONLY place we define the order.
# Add your algs here in the order you want the colors.
GLOBAL_ALG_COLOR_ORDER = [
    "FedAvg",
    "FedProx",
    "SCAFFOLD",
    "Per-FedAvg",
    "pFedMe",
    "FedBABU",
    "MAPL",      # covers MAPL_S-* and MAPL_C-*
    "FedMD",
    "pFedCK",
    "COMET",
    "Ditto",

]

def _canon_algo_name(s: str) -> str:
    """Normalize different text versions to ONE key used in the color map."""
    t = (s or "").strip()
    tl = t.lower().replace(" ", "")
    if "mapl" in tl:
        return "MAPL"
    if "pfedme" in tl:
        return "pFedMe"
    if "pfedck" in tl:
        return "pFedCK"
    return t

# build once
_cmap = plt.get_cmap("tab10")
GLOBAL_ALG_COLOR_MAP: Dict[str, Any] = {}
for i, name in enumerate(GLOBAL_ALG_COLOR_ORDER):
    GLOBAL_ALG_COLOR_MAP[name] = _cmap(i % 10)

def _get_color_for_label(label: str) -> Any:
    """
    label is usually algorithm_display (e.g. 'FedAvg', 'MAPL_C-AlexNet', ...).
    We first try to get the underlying canonical algorithm and then map to color.
    """
    if not label:
        return None
    # try to pull MAPL base name
    base = _canon_algo_name(label)
    if base in GLOBAL_ALG_COLOR_MAP:
        return GLOBAL_ALG_COLOR_MAP[base]
    # last resort: assign a deterministic color based on hash, but
    # try not to get here.
    idx = abs(hash(base)) % 10
    return _cmap(idx)

# ---------------------------------------------------------------------
# 3) LINESTYLES
# ---------------------------------------------------------------------
_mapl_label_re = re.compile(r"^MAPL_(?P<variant>[CS])-(?P<x>.+)$")
def _extract_mapl_x(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label); return m.group("x") if m else None
def _extract_mapl_variant(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label); return m.group("variant") if m else None

def _linestyle_for(label: str) -> str:
    v = _extract_mapl_variant(label)
    if v == "S":
        return "-"
    if v == "C":
        return "--"
    return "-"

# ---------------------------------------------------------------------
# 4) WINDOWS LONG-PATH HELPERS  (unchanged)
# ---------------------------------------------------------------------
def _win_long_abs(path: Path) -> str:
    s = str(path if path.is_absolute() else (Path.cwd() / path))
    if os.name == "nt":
        s = os.path.normpath(s)
        if s.startswith("\\\\?\\") or s.startswith("\\\\?\\UNC\\"):
            return s
        if s.startswith("\\\\"):
            s = "\\\\?\\UNC\\" + s.lstrip("\\")
        else:
            s = "\\\\?\\" + s
    return s

def _open_json_win_safe(p: Path):
    return open(_win_long_abs(p), "r", encoding="utf-8")

def _savefig_longpath(fig, outfile: Path):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(_win_long_abs(outfile), bbox_inches="tight")
    except Exception:
        fig.savefig(str(outfile), bbox_inches="tight")

# ---------------------------------------------------------------------
# 5) JSON WALKERS / FINDERS  (your code — unchanged except for context)
# ---------------------------------------------------------------------
def walk(obj: Json, path: Tuple[str, ...] = ()):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (f"[{i}]",))

def path_str(path: Optional[Tuple[str, ...]]):
    return ".".join(path) if path else "N/A"

# ... (KEEP all your find_* functions exactly as is) ...
# I'm not removing them here — they stay the same.
# ---------------------------------------------------------------------
# 6) LOADING — KEEP your implementation (I just shorten here for brevity)
# ---------------------------------------------------------------------
#  -- paste your full load_rows_from_dir, _load_any_jsons_under, etc. here --
# (UNCHANGED from what you sent)

# ---------------------------------------------------------------------
# 7) CONCAT helper
# ---------------------------------------------------------------------
def _concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=[
            "dataset",
            "algorithm_display",
            "measure",
            "seed",
            "client_id",
            "iteration",
            "accuracy",
            "client_net_type_name",
            "client_net_type_value",
            "alpha_dich",
            "lambda_ditto",
            "_path",
            "algorithm",
        ]
    )

# ---------------------------------------------------------------------
# 8) PLOTTING FUNCTIONS — only change: use _get_color_for_label(...)
# ---------------------------------------------------------------------
def figure_diff_benchmarks(figset_dir: Path, out_root: Path, *, alpha_value: int, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return
    subfigs = [d for d in sorted(figset_dir.iterdir()) if d.is_dir()]
    if not subfigs:
        print(f"[WARN] No subfig folders under {figset_dir}"); return
    loaded = [(sf, _load_subfig_df(sf, inspect)) for sf in subfigs]
    filtered = []
    for sf, df in loaded:
        if df.empty:
            continue
        sub = df[(df["alpha_dich"] == alpha_value)]
        if not sub.empty:
            filtered.append((sf, sub))
    if not filtered:
        print(f"[WARN] No rows for alpha={alpha_value} in {figset_dir}."); return
    filtered = filtered[:4]
    n = len(filtered)

    fig_w = max(5.0 * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.5), sharex=False, sharey=False, squeeze=False)
    axes = axes.flatten()

    for ax, (_, df) in zip(axes, filtered):
        ds_counts = df["dataset"].astype(str).value_counts()
        dataset_choice = "CIFAR100" if "CIFAR100" in ds_counts.index else ds_counts.index[0]
        g = (
            df[df["dataset"].astype(str) == dataset_choice]
            .groupby(["algorithm_display", "iteration"], dropna=False)["accuracy"]
            .mean()
            .reset_index()
            .rename(columns={"accuracy": "avg_accuracy"})
            .sort_values(["algorithm_display", "iteration"])
        )
        for lab in sorted(g["algorithm_display"].astype(str).unique().tolist()):
            sub_lab = g[g["algorithm_display"].astype(str) == lab]
            ax.plot(
                sub_lab["iteration"],
                sub_lab["avg_accuracy"],
                marker=None,
                linewidth=2.0,
                label=str(lab),
                color=_get_color_for_label(lab),
                linestyle=_linestyle_for(lab),
            )
        ax.set_title(dataset_choice)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)

    h, l = _legend_from_axes_row(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)), frameon=False, bbox_to_anchor=(0.5, 0.98))
    outfile = out_root / f"{figset_dir.name}.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

def figure_diff_clients_nets(figset_dir: Path, out_root: Path, *, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return
    subfigs = [d for d in sorted(figset_dir.iterdir()) if d.is_dir()]
    if not subfigs:
        print(f"[WARN] No subfig folders under {figset_dir}"); return
    loaded = [(sf, _load_subfig_df(sf, inspect)) for sf in subfigs]
    loaded = [(sf, df) for sf, df in loaded if not df.empty]
    if not loaded:
        print(f"[WARN] No data for client_net_type row figure."); return

    def _prefer_cifar(df: pd.DataFrame) -> pd.DataFrame:
        mask = df["dataset"].astype(str).str.lower() == "cifar100"
        return df[mask] if mask.any() else df

    loaded = [(sf, _prefer_cifar(df)) for sf, df in loaded if not _prefer_cifar(df).empty]
    if not loaded:
        print(f"[WARN] No usable rows (after CIFAR100 preference) for {figset_dir}."); return
    loaded = loaded[:4]
    n = len(loaded)

    # global y-limits
    all_grp = []
    for _, df in loaded:
        g = (
            df.groupby(["algorithm_display", "iteration"], dropna=False)["accuracy"]
            .mean()
            .reset_index()
            .rename(columns={"accuracy": "avg_accuracy"})
        )
        all_grp.append(g)
    big = pd.concat(all_grp, ignore_index=True)
    ymin = float(big["avg_accuracy"].min())
    ymax = float(big["avg_accuracy"].max())

    fig_w = max(5.0 * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.5), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, (_, df) in zip(axes, loaded):
        g = (
            df.groupby(["algorithm_display", "iteration"], dropna=False)["accuracy"]
            .mean()
            .reset_index()
            .rename(columns={"accuracy": "avg_accuracy"})
            .sort_values(["algorithm_display", "iteration"])
        )
        dom_net = df["client_net_type_name"].astype(str).value_counts().index[0] if "client_net_type_name" in df.columns and not df.empty else "clients"
        dom_net_disp = TITLE_MAP.get(dom_net, dom_net)

        for lab in sorted(g["algorithm_display"].astype(str).unique().tolist()):
            sub_lab = g[g["algorithm_display"].astype(str) == lab]
            ax.plot(
                sub_lab["iteration"],
                sub_lab["avg_accuracy"],
                marker=None,
                linewidth=2.0,
                label=str(lab),
                color=_get_color_for_label(lab),
                linestyle=_linestyle_for(lab),
            )
        ax.set_title(f"{dom_net_disp}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)
        ax.set_ylim(ymin, ymax)

    h, l = _legend_from_axes_row(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)), frameon=False, bbox_to_anchor=(0.5, 0.98))
    outfile = out_root / f"{figset_dir.name}.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

def figure_by_client_net_type_value(figset_dir: Path, out_root: Path, *, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}"); return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    if "client_net_type_value" not in df_all.columns:
        df_all["client_net_type_value"] = None
    if "client_net_type_name" not in df_all.columns:
        df_all["client_net_type_name"] = None

    facet_key = df_all["client_net_type_value"].astype(str)
    bad = facet_key.isna() | (facet_key.str.lower().isin(["none", "nan", ""]))
    facet_key = facet_key.mask(bad, df_all["client_net_type_name"].astype(str))

    values = [v for v in facet_key.unique().tolist() if v and v.lower() not in {"none", "nan"}]
    if not values:
        print(f"[WARN] No usable client_net_type_value/name to facet in {figset_dir}."); return
    values = values[:4]
    n = len(values)

    big = (
        df_all.assign(__facet_key__=facet_key)
        .groupby(["__facet_key__", "algorithm_display", "iteration"], dropna=False)["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "avg_accuracy"})
    )
    ymin = float(big["avg_accuracy"].min())
    ymax = float(big["avg_accuracy"].max())

    fig_w = max(5.0 * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.5), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, val in zip(axes, values):
        sub = df_all[facet_key == val]
        if sub.empty:
            ax.set_visible(False)
            continue

        name_mode = sub["client_net_type_name"].astype(str).value_counts().index[0] if "client_net_type_name" in sub.columns and not sub.empty else None
        value_mode = sub["client_net_type_value"].astype(str).value_counts().index[0] if "client_net_type_value" in sub.columns and not sub.empty else None
        title_name = _net_title_from_map(name_mode, value_mode, val)

        g = (
            sub.groupby(["algorithm_display", "iteration"], dropna=False)["accuracy"]
            .mean()
            .reset_index()
            .rename(columns={"accuracy": "avg_accuracy"})
            .sort_values(["algorithm_display", "iteration"])
        )
        for lab in sorted(g["algorithm_display"].astype(str).unique().tolist()):
            sub_lab = g[g["algorithm_display"].astype(str) == lab]
            ax.plot(
                sub_lab["iteration"],
                sub_lab["avg_accuracy"],
                marker=None,
                linewidth=2.0,
                label=str(lab),
                color=_get_color_for_label(lab),
                linestyle=_linestyle_for(lab),
            )

        ax.set_title(title_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)
        ax.set_ylim(ymin, ymax)

    h, l = _legend_from_axes_row(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)), frameon=False, bbox_to_anchor=(0.5, 0.98))

    outfile = out_root / "client_net_type_value.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

def figure_examine_lambda_for_ditto(figset_dir: Path, out_root: Path, *, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return
    frames: List[pd.DataFrame] = []
    frames.append(load_rows_from_dir(figset_dir, inspect=inspect, alg_hint=None))
    for sub in sorted([d for d in figset_dir.iterdir() if d.is_dir()]):
        frames.append(load_rows_from_dir(sub, inspect=inspect, alg_hint=sub.name))
    df = _concat_or_empty(frames)
    if df.empty:
        print(f"[WARN] No data in {figset_dir}"); return
    df = df[~df["lambda_ditto"].isna()]
    if df.empty:
        print(f"[WARN] No lambda_ditto values found in {figset_dir}"); return
    grp = (
        df.groupby(["lambda_ditto", "iteration"], dropna=False)["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "avg_accuracy"})
    )
    lambdas = sorted(grp["lambda_ditto"].dropna().unique().tolist())
    # this one is about lambda values, not algs, so we can still
    # use tab10 locally here
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
    for i, lmbda in enumerate(lambdas):
        gsub = grp[grp["lambda_ditto"] == lmbda].sort_values("iteration")
        lab = f"λ={lmbda:g}"
        ax.plot(
            gsub["iteration"],
            gsub["avg_accuracy"],
            linewidth=2.0,
            label=lab,
            color=cmap(i % 10),
        )
    ax.set_title("Ditto: effect of λ")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Accuracy (across clients)")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=min(6, len(lambdas)), frameon=False)
    outfile = out_root / f"{figset_dir.name}.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

# ---------------------------------------------------------------------
# 9) CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"))
    ap.add_argument("--figdir", type=Path, default=Path("figures"))
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--lambda_folder", type=str, default="examine_lambda_for_ditto")
    ap.add_argument("--alpha", type=int, default=5)
    ap.add_argument(
        "--facet_client_net_dir",
        type=Path,
        default=Path("results/diff_clients_nets"),
        help="(kept for compatibility; not used directly)"
    )
    args = ap.parse_args()

    figure_diff_clients_nets(args.root / "diff_clients_nets", args.figdir, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "diff_benchmarks_05", args.figdir, alpha_value=args.alpha, inspect=args.inspect)
    figure_by_client_net_type_value(args.root / "same_client_nets", args.figdir, inspect=args.inspect)
    figure_examine_lambda_for_ditto(args.root / args.lambda_folder, args.figdir, inspect=args.inspect)

if __name__ == "__main__":
    main()
