#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_algorithms_avg_accuracy.py

This builds ONE figure per figset:

results/
  diff_clients_nets/                 # FIGSET 1 (one figure; 1xN subplots)
    <subfigureA>/
      <algorithm1>/*.json
      <algorithm2>/*.json
    <subfigureB>/ ...
  diff_benchmarks_05/                # FIGSET 2 (one figure; 1xN subplots; alpha filter)
    <subfigureA>/
      <algorithm1>/*.json
      <algorithm2>/*.json
    <subfigureB>/ ...

Outputs (one PDF per figset):
figures/<FIGSET>/<FIGSET>.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

TITLE_MAP = {
    "rndNet":    "{Alex, Res, Mobile, Squeeze}",
    "rndStrong": "{Alex, Res}",
    "rndWeak":   "{Mobile, Squeeze}",
}
def _right_of_dash(s: str) -> str:
    # split on hyphen or en/em dash; keep the part after it
    parts = re.split(r"\s*[-–—]\s*", str(s), maxsplit=1)
    return parts[1] if len(parts) > 1 else str(s)
# --------------------- Windows long-path helpers ---------------------

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

# --------------------- tiny utils ---------------------

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

# --------------------- field finders ---------------------

def find_algorithm_name(root: Json):
    c: List[Tuple[Tuple[str, ...], str]] = []
    for p, v in walk(root):
        if isinstance(v, str) and v.strip():
            if p and p[-1].lower() in {"_name_", "name"} and any("algorithm" in seg.lower() for seg in p):
                c.append((p, v.strip()))
            if p and p[-1].lower() == "algorithm":
                c.append((p, v.strip()))
    if c:
        def score(path: Tuple[str, ...]) -> int:
            s = 0
            if path and path[-1].lower() in {"_name_", "name"}: s += 2
            if any("algorithm" in seg.lower() for seg in path): s += 2
            if path and path[-1].lower() == "algorithm": s += 3
            return s
        best = max(c, key=lambda t: score(t[0]))
        return best[1], best[0]
    return None, None

def find_seed_num(root: Json):
    for p, v in walk(root):
        if not p: continue
        key = p[-1].lower()
        if key in {"seed_num", "seed"}:
            for caster in (int, lambda x: int(float(x))):
                try: return caster(v), p
                except Exception: pass
    return None, None

def looks_like_client_accuracy_map(v: Any) -> bool:
    if not isinstance(v, dict) or not v: return False
    checked, nested = 0, 0
    for _, inner in v.items():
        checked += 1
        if isinstance(inner, dict) and inner:
            for it_val in inner.values():
                if isinstance(it_val, (int, float, str)):
                    nested += 1
                    break
        if checked >= 5: break
    return nested > 0

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
    if v is not None: return v, p
    c: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if looks_like_client_accuracy_map(v): c.append((p, v))
    if c:
        def score(pv):
            p,_ = pv; s=0
            if any("summary" in seg.lower() for seg in p): s+=2
            if any("client" in seg.lower() for seg in p): s+=1
            return s
        c.sort(key=score, reverse=True)
        return c[0][1], c[0][0]
    return None, None

def find_server_accuracy_map(root: Json):
    v, p = find_exact_key_map(root, "server_accuracy_per_client_1_max")
    if v is not None: return v, p
    c: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if looks_like_client_accuracy_map(v) and any("server" in seg.lower() for seg in p):
            c.append((p, v))
    if c:
        def score(pv):
            p,_=pv; s=0
            if any("summary" in seg.lower() for seg in p): s+=2
            if any("server" in seg.lower() for seg in p): s+=1
            return s
        c.sort(key=score, reverse=True)
        return c[0][1], c[0][0]
    return None, None

def find_server_net_type_value(root: Json):
    best: Optional[Tuple[Tuple[str, ...], str]] = None
    best_score = -1
    for p, v in walk(root):
        if not p: continue
        if p[-1] in {"_value_", "value"} and isinstance(v, (str, int, float)):
            pl = [seg.lower() for seg in p]
            if any("server" in seg for seg in pl) and any("net" in seg for seg in pl):
                score = 1 + (2 if any("summary" in seg for seg in pl) else 0)
                if score > best_score:
                    best = (p, str(v)); best_score = score
    if best: return best[1], best[0]
    return None, None

def find_client_net_type_name(root: Json):
    # Prefer summary.client_net_type_name if present
    cur = root
    try:
        if isinstance(cur, dict) and "summary" in cur:
            cur = cur["summary"]
            if isinstance(cur, dict) and "client_net_type_name" in cur:
                v = cur["client_net_type_name"]
                return (str(v) if v is not None else None), ("summary","client_net_type_name")
    except Exception:
        pass
    # Fallback: anywhere
    for p, v in walk(root):
        if p and p[-1] == "client_net_type_name":
            return (str(v) if v is not None else None), p
    # Fallback 2: summary.client_net_type._name_ / _value_
    try:
        cur = root
        if isinstance(cur, dict) and "summary" in cur and isinstance(cur["summary"], dict):
            s = cur["summary"]
            if isinstance(s.get("client_net_type"), dict):
                d = s["client_net_type"]
                for k in ("_name_","_value_"):
                    if k in d and d[k] is not None:
                        return str(d[k]), ("summary","client_net_type",k)
    except Exception:
        pass
    return None, None

def find_dataset_selected_name(root: Json):
    cur = root
    try:
        if isinstance(cur, dict) and "summary" in cur:
            cur = cur["summary"]
            if isinstance(cur, dict) and "data_set_selected" in cur:
                cur2 = cur["data_set_selected"]
                if isinstance(cur2, dict) and "_name_" in cur2:
                    v = cur2["_name_"]
                    return (str(v) if v is not None else None), ("summary","data_set_selected","_name_")
    except Exception:
        pass
    for p, v in walk(root):
        if not p or not isinstance(v, (str,int,float)): continue
        pl = [seg.lower() for seg in p]
        if any("summary" in seg for seg in pl) and any("data_set_selected" in seg for seg in pl):
            if p[-1].lower() in {"_name_","name"}: return str(v), p
    return None, None

def find_alpha_dich(root: Json):
    # Prefer summary.alpha_dich
    try:
        if isinstance(root, dict) and "summary" in root:
            s = root["summary"]
            if isinstance(s, dict) and "alpha_dich" in s:
                v = s["alpha_dich"]
                for caster in (int, lambda x: int(float(x))):
                    try: return caster(v), ("summary","alpha_dich")
                    except Exception: pass
                return None, ("summary","alpha_dich")
    except Exception:
        pass
    # Fallback: anywhere
    for p, v in walk(root):
        if p and p[-1] == "alpha_dich":
            for caster in (int, lambda x: int(float(x))):
                try: return caster(v), p
                except Exception: pass
            return None, p
    return None, None

# --------------------- loading ---------------------

def load_rows_from_dir(dir_path: Path, *, inspect: bool = False, alg_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Build rows ONLY from JSONs under dir_path (single algorithm folder).
    If a JSON lacks an algorithm name, fall back to `alg_hint` (usually the folder name).
    """
    rows: List[Dict[str, Any]] = []
    files_scanned = 0
    files_used = 0

    json_paths = list(dir_path.glob("*.json")) + list(dir_path.glob("*.JSON"))
    if inspect:
        print(f"\n[SCAN] {dir_path} -> {len(json_paths)} JSONs (alg_hint={alg_hint!r})")
        for jp in json_paths:
            print(" -", jp.name)

    for p in sorted(json_paths):
        files_scanned += 1
        try:
            with _open_json_win_safe(p) as fh:
                raw = json.load(fh)
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}")
            continue

        ds_name, ds_path = find_dataset_selected_name(raw)
        alg_name, alg_path = find_algorithm_name(raw)
        if not alg_name:
            alg_name = alg_hint or dir_path.name  # fallback to folder name

        seed_num, seed_path = find_seed_num(raw)
        client_map, client_path = find_client_accuracy_map_anywhere(raw)
        server_map, server_path = find_server_accuracy_map(raw)
        server_net_val, server_net_path = find_server_net_type_value(raw)
        client_net_name, client_net_path = find_client_net_type_name(raw)
        alpha_dich, alpha_path = find_alpha_dich(raw)

        if inspect:
            print(f"[INSPECT] {p.name}")
            print(f"  dataset_selected._name_: {ds_name} @ {path_str(ds_path)}")
            print(f"  algorithm: {alg_name} @ {path_str(alg_path) if alg_path else '(folder fallback)'}")
            print(f"  seed_num : {seed_num} @ {path_str(seed_path)}")
            print(f"  client_accuracy_per_client_1: @ {path_str(client_path)}  (found={client_map is not None})")
            print(f"  server_accuracy_per_client_1_max: @ {path_str(server_path)} (found={server_map is not None})")
            print(f"  server_net_type._value_: {server_net_val} @ {path_str(server_net_path)}")
            print(f"  client_net_type_name: {client_net_name} @ {path_str(client_net_path)}")
            print(f"  alpha_dich: {alpha_dich} @ {path_str(alpha_path)}")

        is_mapl = bool(alg_name and "mapl" in str(alg_name).lower())
        used_any = False

        # client measure (all algs)
        if isinstance(client_map, dict) and client_map:
            display_label = str(alg_name)
            if is_mapl:
                x = server_net_val if server_net_val is not None else "NA"
                display_label = f"MAPL_C-{x}"

            for client_id, iter_dict in client_map.items():
                iter_pairs = iter_dict.items() if isinstance(iter_dict, dict) else client_map.items()
                if not isinstance(iter_dict, dict):
                    client_id = "ALL"
                for iter_k, acc_v in iter_pairs:
                    try:
                        iteration = int(iter_k)
                    except Exception:
                        try: iteration = int(float(iter_k))
                        except Exception: continue
                    try:
                        acc = float(acc_v)
                    except Exception:
                        continue
                    rows.append({
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
                    })
                    used_any = True

        # server measure (MAPL only)
        if is_mapl and isinstance(server_map, dict) and server_map:
            x = server_net_val if server_net_val is not None else "NA"
            display_label = f"MAPL_S-{x}"
            for client_id, iter_dict in server_map.items():
                iter_pairs = iter_dict.items() if isinstance(iter_dict, dict) else server_map.items()
                if not isinstance(iter_dict, dict):
                    client_id = "ALL"
                for iter_k, acc_v in iter_pairs:
                    try:
                        iteration = int(iter_k)
                    except Exception:
                        try: iteration = int(float(iter_k))
                        except Exception: continue
                    try:
                        acc = float(acc_v)
                    except Exception:
                        continue
                    rows.append({
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
                    })
                    used_any = True

        if used_any:
            files_used += 1

    if inspect:
        print(f"[SUMMARY] scanned={files_scanned}, used={files_used} in {dir_path}")

    cols = [
        "dataset","algorithm_display","measure","seed","client_id",
        "iteration","accuracy","client_net_type_name","alpha_dich","_path","algorithm"
    ]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# --------------------- styling helpers ---------------------

_mapl_label_re = re.compile(r"^MAPL_(?P<variant>[CS])-(?P<x>.+)$")
def _extract_mapl_x(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label); return m.group("x") if m else None
def _extract_mapl_variant(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label); return m.group("variant") if m else None

def _build_color_map(labels: List[str]):
    cmap = plt.get_cmap("tab10")
    mapl_x_values, non_mapl = [], []
    for lab in labels:
        x = _extract_mapl_x(lab)
        (mapl_x_values if x is not None else non_mapl).append(x or lab)
    unique_x = list(dict.fromkeys([x for x in mapl_x_values]))  # preserve order
    color_by_x = {x: cmap(i % 10) for i, x in enumerate(unique_x)}
    color_by_label: Dict[str, Any] = {}
    for lab in labels:
        x = _extract_mapl_x(lab)
        if x is not None: color_by_label[lab] = color_by_x[x]
    next_idx = len(unique_x)
    for lab in labels:
        if lab not in color_by_label:
            color_by_label[lab] = cmap(next_idx % 10)
            next_idx += 1
    return color_by_label

def _linestyle_for(label: str) -> str:
    v = _extract_mapl_variant(label)
    if v == "S": return "-"
    if v == "C": return "--"
    return "-"

# --------------------- helpers for combined figures ---------------------

def _concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["dataset","algorithm_display","measure","seed","client_id",
                 "iteration","accuracy","client_net_type_name","alpha_dich","_path","algorithm"]
    )

def _load_subfig_df(subfig_dir: Path, inspect: bool) -> pd.DataFrame:
    """Load and concat all algorithm folders under one subfig directory into a single DataFrame."""
    alg_dirs = [a for a in sorted(subfig_dir.iterdir()) if a.is_dir()]
    frames = []
    for alg_dir in alg_dirs:
        frames.append(load_rows_from_dir(alg_dir, inspect=inspect, alg_hint=alg_dir.name))
    return _concat_or_empty(frames)

def _legend_from_axes_row(axes):
    handles, labels = [], []
    for a in axes:
        h, l = a.get_legend_handles_labels()
        handles += h; labels += l
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    return [h for h,_ in uniq], [l for _,l in uniq]

# --------------------- plotting (ONE FIGURE per FIGSET) ---------------------

def figure_diff_benchmarks(figset_dir: Path, out_root: Path, *, alpha_value: int, inspect: bool):
    """One ROW, each subplot is a subfig folder; independent Y axes, one legend."""
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    subfigs = [d for d in sorted(figset_dir.iterdir()) if d.is_dir()]
    if not subfigs:
        print(f"[WARN] No subfig folders under {figset_dir}"); return

    # Load all once
    loaded = [(sf, _load_subfig_df(sf, inspect)) for sf in subfigs]
    # Filter & keep only non-empty after filtering condition
    filtered = []
    for sf, df in loaded:
        if df.empty:
            continue
        sub = df[
                 (df["alpha_dich"] == alpha_value)]
        if not sub.empty:
            filtered.append((sf, sub))
    if not filtered:
        print(f"[WARN] No rows for rndWeak & alpha={alpha_value} in {figset_dir}."); return

    # Limit to 4 subplots max (as requested)
    filtered = filtered[:4]
    n = len(filtered)

    # Build labels/colors globally (consistent colors across panels)
    all_labels = sorted(pd.concat([x[1] for x in filtered])["algorithm_display"].astype(str).unique().tolist())
    color_by_label = _build_color_map(all_labels)

    fig_w = max(5.0 * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.5), sharex=False, sharey=False, squeeze=False)
    axes = axes.flatten()

    for ax, (sf, df) in zip(axes, filtered):
        # If multiple datasets are present within this subfig, prefer CIFAR100; else use the most frequent dataset
        ds_counts = df["dataset"].astype(str).value_counts()
        dataset_choice = "CIFAR100" if "CIFAR100" in ds_counts.index else ds_counts.index[0]

        g = (
            df[df["dataset"].astype(str) == dataset_choice]
            .groupby(["algorithm_display","iteration"], dropna=False)["accuracy"]
            .mean()
            .reset_index()
            .rename(columns={"accuracy":"avg_accuracy"})
            .sort_values(["algorithm_display","iteration"])
        )

        for lab in sorted(g["algorithm_display"].astype(str).unique().tolist()):
            sub_lab = g[g["algorithm_display"].astype(str) == lab]
            ax.plot(sub_lab["iteration"], sub_lab["avg_accuracy"],
                    marker=None, linewidth=2.0, label=str(lab),
                    color=color_by_label.get(lab, None), linestyle=_linestyle_for(lab))
        ax.set_title(f"{dataset_choice}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)

    # One legend
    h, l = _legend_from_axes_row(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)), frameon=False, bbox_to_anchor=(0.5, 0.98))

    outfile = out_root /  f"{figset_dir.name}.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

def figure_diff_clients_nets(figset_dir: Path, out_root: Path, *, inspect: bool):
    """One ROW, each subplot is a subfig folder; SHARED Y axis, one legend."""
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    subfigs = [d for d in sorted(figset_dir.iterdir()) if d.is_dir()]
    if not subfigs:
        print(f"[WARN] No subfig folders under {figset_dir}"); return

    # Load all once
    loaded = [(sf, _load_subfig_df(sf, inspect)) for sf in subfigs]
    loaded = [(sf, df) for sf, df in loaded if not df.empty]
    if not loaded:
        print(f"[WARN] No data for client_net_type row figure."); return

    # Prefer CIFAR100 rows for consistency
    def _prefer_cifar(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["dataset"].astype(str).str.lower() == "cifar100"] if (df["dataset"].astype(str).str.lower() == "cifar100").any() else df

    loaded = [(sf, _prefer_cifar(df)) for sf, df in loaded if not _prefer_cifar(df).empty]
    if not loaded:
        print(f"[WARN] No usable rows (after CIFAR100 preference) for {figset_dir}."); return

    # Limit to 4 subplots max (as requested)
    loaded = loaded[:4]
    n = len(loaded)

    # Global colors by algorithm (consistent across subplots)
    all_labels = sorted(pd.concat([x[1] for x in loaded])["algorithm_display"].astype(str).unique().tolist())
    color_by_label = _build_color_map(all_labels)

    fig_w = max(5.0 * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.5), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    # Compute global y-limits (shared y)
    all_grp = []
    for _, df in loaded:
        g = (df.groupby(["algorithm_display","iteration"], dropna=False)["accuracy"]
                .mean().reset_index().rename(columns={"accuracy":"avg_accuracy"}))
        all_grp.append(g)
    big = pd.concat(all_grp, ignore_index=True)
    ymin = float(big["avg_accuracy"].min())
    ymax = float(big["avg_accuracy"].max())

    for ax, (sf, df) in zip(axes, loaded):
        g = (
            df.groupby(["algorithm_display","iteration"], dropna=False)["accuracy"]
              .mean().reset_index().rename(columns={"accuracy":"avg_accuracy"})
              .sort_values(["algorithm_display","iteration"])
        )
        # Optional: if you want to show the client_net facet in the title (not faceting inside),
        # show the dominant client_net_type_name in this subfig
        dom_net = df["client_net_type_name"].astype(str).value_counts().index[0] if "client_net_type_name" in df.columns and not df.empty else "clients"
        dom_net_disp = TITLE_MAP.get(dom_net, dom_net)

        for lab in sorted(g["algorithm_display"].astype(str).unique().tolist()):
            sub_lab = g[g["algorithm_display"].astype(str) == lab]
            ax.plot(sub_lab["iteration"], sub_lab["avg_accuracy"],
                    marker=None, linewidth=2.0, label=str(lab),
                    color=color_by_label.get(lab, None), linestyle=_linestyle_for(lab))
        ax.set_title(f"{dom_net_disp}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)
        ax.set_ylim(ymin, ymax)

    # One legend
    h, l = _legend_from_axes_row(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)), frameon=False, bbox_to_anchor=(0.5, 0.98))

    outfile = out_root /  f"{figset_dir.name}.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"), help="Root results folder (contains figure-set folders)")
    ap.add_argument("--figdir", type=Path, default=Path("figures"), help="Where to save PDFs")

    ap.add_argument("--inspect", action="store_true", help="Print detected paths/keys while loading")

    args = ap.parse_args()

    figure_diff_clients_nets(args.root / "diff_clients_nets", args.figdir, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "diff_benchmarks_05", args.figdir, alpha_value=5, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "same_client_nets", args.figdir, alpha_value=5, inspect=args.inspect)

if __name__ == "__main__":
    main()
