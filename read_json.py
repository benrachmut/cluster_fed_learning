#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_algorithms_avg_accuracy.py  (MAPL_C-X + MAPL_S-X with shared colors by X)

Figure:
- x-axis: iteration
- y-axis: average accuracy across ALL clients & ALL seeds
- one line per displayed label:
    * Non-MAPL: algorithm name (solid)
    * MAPL:
        - MAPL_C-<X>  (client_accuracy_per_client_1)  -> dashed
        - MAPL_S-<X>  (server_accuracy_per_client_1_max) -> solid
      where <X> = summary.server_net_type._value_
    * MAPL_C-X and MAPL_S-X share the **same color** for the same X.

Usage:
  python make_fig_algorithms_avg_accuracy.py --root results --inspect
  python make_fig_algorithms_avg_accuracy.py --root results --figdir figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# ---------- Recursive traversal ----------
def walk(obj: Json, path: Tuple[str, ...] = ()):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (f"[{i}]",))

def path_str(path: Optional[Tuple[str, ...]]) -> str:
    return ".".join(path) if path else "N/A"

# ---------- Heuristics to detect fields ----------
def find_algorithm_name(root: Json) -> Tuple[Optional[str], Optional[Tuple[str, ...]]]:
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

def find_seed_num(root: Json) -> Tuple[Optional[int], Optional[Tuple[str, ...]]]:
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

def find_exact_key_map(root: Json, keyname: str) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[Tuple[str, ...]]]:
    hits: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if p and p[-1] == keyname and isinstance(v, dict):
            hits.append((p, v))
    if hits:
        hits.sort(key=lambda t: len(t[0]))  # shortest path
        return hits[0][1], hits[0][0]
    return None, None

def find_client_accuracy_map_anywhere(root: Json) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[Tuple[str, ...]]]:
    v, p = find_exact_key_map(root, "client_accuracy_per_client_1")
    if v is not None:
        return v, p
    # Heuristic fallback
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

def find_server_accuracy_map(root: Json) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[Tuple[str, ...]]]:
    v, p = find_exact_key_map(root, "server_accuracy_per_client_1_max")
    if v is not None:
        return v, p
    # Heuristic fallback
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

def find_server_net_type_value(root: Json) -> Tuple[Optional[str], Optional[Tuple[str, ...]]]:
    # Preferred: summary.server_net_type._value_
    for p, v in walk(root):
        if not p:
            continue
        if p[-1] in {"_value_", "value"} and isinstance(v, (str, int, float)):
            if any(seg.lower() == "server_net_type" for seg in p) and any(seg.lower() == "summary" for seg in p):
                return str(v), p
    # Heuristic fallback
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

# ---------- Loading ----------
def load_rows(root: Path, inspect: bool = False) -> pd.DataFrame:
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

        alg_name, alg_path = find_algorithm_name(raw)
        seed_num, seed_path = find_seed_num(raw)
        client_map, client_path = find_client_accuracy_map_anywhere(raw)
        server_map, server_path = find_server_accuracy_map(raw)
        server_net_val, server_net_path = find_server_net_type_value(raw)

        if inspect:
            print(f"\n--- Inspect: {p}")
            print(f"  algorithm: {alg_name} @ {path_str(alg_path)}")
            print(f"  seed_num : {seed_num} @ {path_str(seed_path)}")
            print(f"  client_accuracy_per_client_1: @ {path_str(client_path)}")
            print(f"  server_accuracy_per_client_1_max: @ {path_str(server_path)}")
            print(f"  server_net_type._value_: {server_net_val} @ {path_str(server_net_path)}")

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
                            "algorithm": alg_name or "unknown_alg",
                            "algorithm_display": display_label,
                            "measure": "client",  # client-side measure
                            "seed": seed_num,
                            "client_id": str(client_id),
                            "iteration": iteration,
                            "accuracy": acc,
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
                            "algorithm": alg_name or "unknown_alg",
                            "algorithm_display": display_label,
                            "measure": "server",  # server-side measure
                            "seed": seed_num,
                            "client_id": str(client_id),
                            "iteration": iteration,
                            "accuracy": acc,
                        }
                    )
                    used_any = True

        if used_any:
            files_used += 1

    if inspect:
        print(f"\nScanned {files_scanned} JSON files; usable for plot: {files_used}")

    if not rows:
        return pd.DataFrame(columns=["algorithm_display", "measure", "seed", "client_id", "iteration", "accuracy"])
    return pd.DataFrame(rows)

# ---------- Plotting (shared colors for X; styles by C/S) ----------
_mapl_label_re = re.compile(r"^MAPL_(?P<variant>[CS])-(?P<x>.+)$")

def _extract_mapl_x(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label)
    return m.group("x") if m else None

def _extract_mapl_variant(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label)
    return m.group("variant") if m else None

def plot_algorithms_avg_accuracy(df: pd.DataFrame, outdir: Path):
    if df.empty:
        print("[WARN] No usable rows to plot.")
        return

    # Average across clients & seeds per (display label, iteration)
    grp = (
        df.groupby(["algorithm_display", "iteration"], dropna=False)["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "avg_accuracy"})
    )

    # Build color map: same X => same color (for MAPL_*), others get their own colors
    labels = sorted(grp["algorithm_display"].unique())
    mapl_x_values = []
    non_mapl_labels = []
    for lab in labels:
        x = _extract_mapl_x(lab)
        if x is not None:
            mapl_x_values.append(x)
        else:
            non_mapl_labels.append(lab)
    unique_x = list(dict.fromkeys(mapl_x_values))  # preserve order

    cmap = plt.get_cmap("tab10")
    color_by_x: Dict[str, Any] = {x: cmap(i % 10) for i, x in enumerate(unique_x)}

    # Assign colors: MAPL_* -> by X; others -> from remaining cycle positions
    color_by_label: Dict[str, Any] = {}
    next_color_idx = 0
    for lab in labels:
        x = _extract_mapl_x(lab)
        if x is not None:
            color_by_label[lab] = color_by_x[x]
        else:
            color_by_label[lab] = cmap((next_color_idx + len(unique_x)) % 10)
            next_color_idx += 1

    # Line styles: S=solid, C=dashed, others=solid
    def linestyle_for(label: str) -> str:
        v = _extract_mapl_variant(label)
        if v == "S":
            return "-"   # solid
        if v == "C":
            return "--"  # not solid
        return "-"       # default solid for non-MAPL

    # Plot
    plt.figure()
    for lab in labels:
        sub = grp[grp["algorithm_display"] == lab].sort_values("iteration")
        plt.plot(
            sub["iteration"],
            sub["avg_accuracy"],
            marker="o",
            label=str(lab),
            color=color_by_label[lab],
            linestyle=linestyle_for(lab),
        )

    plt.xlabel("Iteration")
    plt.ylabel("Average Accuracy (across clients & seeds)")
    plt.title("Algorithms: Average Accuracy vs Iteration (MAPL_C/S share colors by X)")
    plt.legend()
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "algorithms_avg_accuracy.pdf"
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure: {outfile}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"), help="Root results folder")
    ap.add_argument("--figdir", type=Path, default=Path("figures"), help="Where to save the PDF")
    ap.add_argument("--inspect", action="store_true", help="Print per-file detected paths/keys")
    args = ap.parse_args()

    df = load_rows(args.root, inspect=args.inspect)
    print(f"Loaded rows: {len(df):,}")
    if not df.empty:
        print(df.head(10).to_string(index=False))

    if not args.inspect:
        plot_algorithms_avg_accuracy(df, args.figdir)

if __name__ == "__main__":
    main()
