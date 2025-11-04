#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_algorithms_avg_accuracy.py

Adds new figures:
  1) Facets subplots by `client_net_type._value_` (row of subplots).
  2) Heatmap of final accuracy: Algorithm × Client Net Type.
  3) MAPL/global-data-size: ONE plot (one curve per subfolder, labeled by server_data_ratio).

Existing figsets:
  - diff_clients_nets/       (row of subplots by subfolder)
  - diff_benchmarks_05/      (row of subplots by subfolder; alpha filter)
  - examine_lambda_for_ditto (single figure; curves by λ_ditto)

Outputs:
  - figures/same_client_nets/client_net_type_value.pdf
  - figures/same_client_nets/client_net_type_heatmap.pdf
  - figures/<global_data_size_folder>/<global_data_size_folder>_by_server_data_ratio_{client|server}.pdf
  - figures/diff_clients_nets/diff_clients_nets.pdf
  - figures/diff_benchmarks_05/diff_benchmarks_05.pdf
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
import numpy as np

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# ---------------------------------------------------------------------
# Comparison figures baseline: plot CLIENT curves (all algorithms),
# and ALSO overlay MAPL SERVER curves where available.
# ---------------------------------------------------------------------
PLOT_MEASURE_BASE = "client"         # base measure for comparisons
INCLUDE_SERVER_FOR_MAPL = True       # overlay MAPL server curves in comparison figs

TITLE_MAP = {
    "rndNet":    "{AlexNet, ResNet, MobileNet, SqueezeNet}",
    "rndStrong": "{AlexNet, ResNet}",
    "rndWeak":   "{MobileNet, SqueezeNet}",
}

TITLE_NET_MAP = {
    "ALEXNET": "AlexNet",
    "MobileNet": "MobileNet",
    "RESNET": "ResNet",
    "SqueezeNet": "SqueezeNet",
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

# =====================================================================
# Stable colors per (canonical) algorithm label
# =====================================================================
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
    t = (s or "").strip()
    tl = t.lower().replace(" ", "")
    if "mapl" in tl:
        return "MAPL"
    if "pfedme" in tl:
        return "pFedMe"
    if "pfedck" in tl:
        return "pFedCK"
    return t

GLOBAL_ALG_COLOR_MAP: Dict[str, Any] = {
    "FedAvg":    "#1f77b4",
    "FedProx":   "#ff7f0e",
    "SCAFFOLD":  "#2ca02c",
    "Per-FedAvg":"#d62728",
    "pFedMe":    "#9467bd",
    "FedBABU":   "#8c564b",
    "MAPL":      "#e377c2",
    "FedMD":     "#7f7f7f",
    "pFedCK":    "#bcbd22",
    "COMET":     "#17becf",
    "Ditto":     "#aec7e8",
}

def _get_color_for_label(label: str) -> Any:
    if not label:
        return None
    base = _canon_algo_name(label)
    if base in GLOBAL_ALG_COLOR_MAP:
        return GLOBAL_ALG_COLOR_MAP[base]
    if label in GLOBAL_ALG_COLOR_MAP:
        return GLOBAL_ALG_COLOR_MAP[label]
    return None

# =====================================================================

def _load_any_jsons_under(root_dir: Path, inspect: bool) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    frames.append(load_rows_from_dir(root_dir, inspect=inspect, alg_hint=None))
    for sub in sorted([d for d in root_dir.iterdir() if d.is_dir()]):
        df_alg = load_rows_from_dir(sub, inspect=inspect, alg_hint=sub.name)
        if not df_alg.empty:
            frames.append(df_alg)
        for alg_dir in sorted([d for d in sub.iterdir() if d.is_dir()]):
            df_deep = load_rows_from_dir(alg_dir, inspect=inspect, alg_hint=alg_dir.name)
            if not df_deep.empty:
                frames.append(df_deep)
    return _concat_or_empty(frames)

def _right_of_dash(s: str) -> str:
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
        if isinstance(inner, dict) and inner:
            for it_val in inner.values():
                if isinstance(it_val, (int, float, str)):
                    nested += 1
                    break
        checked += 1
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
    try:
        if isinstance(root, dict) and "summary" in root:
            s = root["summary"]
            if isinstance(s, dict) and "client_net_type_name" in s:
                v = s["client_net_type_name"]
                return (str(v) if v is not None else None), ("summary","client_net_type_name")
    except Exception:
        pass
    for p, v in walk(root):
        if p and p[-1] == "client_net_type_name":
            return (str(v) if v is not None else None), p
    try:
        if isinstance(root, dict) and "summary" in root and isinstance(root["summary"], dict):
            s = root["summary"]
            if isinstance(s.get("client_net_type"), dict):
                d = s["client_net_type"]
                for k in ("_name_","_value_"):
                    if k in d and d[k] is not None:
                        return str(d[k]), ("summary","client_net_type",k)
    except Exception:
        pass
    return None, None

def find_client_net_type_value(root: Json):
    try:
        if isinstance(root, dict) and "summary" in root:
            s = root["summary"]
            if isinstance(s, dict) and isinstance(s.get("client_net_type"), dict):
                d = s["client_net_type"]
                if "_value_" in d and d["_value_"] is not None:
                    return str(d["_value_"]), ("summary","client_net_type","_value_")
    except Exception:
        pass
    for p, v in walk(root):
        if not p: continue
        if p[-1] in {"_value_", "value"} and isinstance(v, (str, int, float)):
            pl = [seg.lower() for seg in p]
            if any("client" in seg for seg in pl) and any("net" in seg for seg in pl):
                return str(v), p
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
        if any("summary" in seg for seg in pl) and any("data_set_selected" in seg for seg in p):
            if p[-1].lower() in {"_name_","name"}: return str(v), p
    return None, None

def find_alpha_dich(root: Json):
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
    for p, v in walk(root):
        if p and p[-1] == "alpha_dich":
            for caster in (int, lambda x: int(float(x))):
                try: return caster(v), p
                except Exception: pass
            return None, p
    return None, None

def find_lambda_ditto(root: Json):
    try:
        if isinstance(root, dict) and "summary" in root:
            s = root["summary"]
            if isinstance(s, dict) and "lambda_ditto" in s:
                v = s["lambda_ditto"]
                try:
                    return float(v), ("summary","lambda_ditto")
                except Exception:
                    try:
                        return float(str(v).strip()), ("summary","lambda_ditto")
                    except Exception:
                        return None, ("summary","lambda_ditto")
    except Exception:
        pass
    for p, v in walk(root):
        if p and p[-1] == "lambda_ditto":
            try:
                return float(v), p
            except Exception:
                try:
                    return float(str(v).strip()), p
                except Exception:
                    return None, p
    return None, None

def find_server_data_ratio(root: Json, *, fallback_from=None):
    def _cast_float(x):
        for caster in (float, lambda y: float(str(y).strip())):
            try: return caster(x)
            except Exception: pass
        return None

    try:
        s = root.get("summary", {}) if isinstance(root, dict) else {}
        for k in ("server_data_ratio", "server_ratio", "server_data_fraction",
                  "server_train_ratio", "server_data_frac"):
            if k in s and s[k] is not None:
                v = _cast_float(s[k])
                if v is not None: return v, ("summary", k)
        if "server_data_pct" in s and s["server_data_pct"] is not None:
            v = _cast_float(s["server_data_pct"])
            if v is not None:
                if v > 1.0: v = v / 100.0
                return v, ("summary", "server_data_pct")
        if "server_data_samples" in s and "total_data_samples" in s:
            a = _cast_float(s["server_data_samples"])
            b = _cast_float(s["total_data_samples"])
            if a is not None and b and b > 0:
                return a / b, ("summary", "server_data_samples/total_data_samples")
    except Exception:
        pass

    for p, v in walk(root):
        if not p: continue
        key = p[-1]
        if key in {"server_data_ratio","server_ratio","server_data_fraction","server_train_ratio","server_data_frac"}:
            val = _cast_float(v)
            if val is not None:
                return val, tuple(p)
        if key == "server_data_pct":
            val = _cast_float(v)
            if val is not None:
                if v > 1.0: val = val / 100.0
                return val, tuple(p)

    if fallback_from:
        s = str(fallback_from)
        for pat in (r"(?:ratio|srv|server)[-_]?(?:data)?[-_]*(?:size|frac|ratio)?[-_]*(\d+(?:\.\d+)?)",
                    r"\br(\d+(?:\.\d+)?)\b",
                    r"pct(\d+(?:\.\d+)?)"):
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                num = m.group(1)
                try:
                    v = float(num)
                    if "pct" in pat and v > 1.0:
                        v = v / 100.0
                    return v, ("path", "regex")
                except Exception:
                    pass

    return None, None

# --------------------- loading ---------------------

def load_rows_from_dir(dir_path: Path, *, inspect: bool = False, alg_hint: Optional[str] = None) -> pd.DataFrame:
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

        ds_name, _           = find_dataset_selected_name(raw)
        alg_name, _          = find_algorithm_name(raw)
        if not alg_name:
            alg_name = alg_hint or dir_path.name
        seed_num, _          = find_seed_num(raw)
        client_map, _        = find_client_accuracy_map_anywhere(raw)
        server_map, _        = find_server_accuracy_map(raw)
        server_net_val, _    = find_server_net_type_value(raw)
        client_net_name, _   = find_client_net_type_name(raw)
        client_net_value, _  = find_client_net_type_value(raw)
        alpha_dich, _        = find_alpha_dich(raw)
        lambda_ditto, _      = find_lambda_ditto(raw)
        server_data_ratio, _ = find_server_data_ratio(raw, fallback_from=p)

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
                        "client_net_type_value": client_net_value or "unknown_client_net",
                        "alpha_dich": alpha_dich,
                        "lambda_ditto": lambda_ditto,
                        "server_data_ratio": server_data_ratio,
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
                        "client_net_type_value": client_net_value or "unknown_client_net",
                        "alpha_dich": alpha_dich,
                        "lambda_ditto": lambda_ditto,
                        "server_data_ratio": server_data_ratio,
                    })
                    used_any = True

        if used_any:
            files_used += 1

    if inspect:
        print(f"[SUMMARY] scanned={files_scanned}, used={files_used} in {dir_path}")

    cols = [
        "dataset","algorithm_display","measure","seed","client_id",
        "iteration","accuracy","client_net_type_name","client_net_type_value",
        "alpha_dich","lambda_ditto","server_data_ratio","_path","algorithm"
    ]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# --------------------- styling helpers ---------------------

_mapl_label_re = re.compile(r"^MAPL_(?P<variant>[CS])-(?P<x>.+)$")
def _extract_mapl_variant(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label); return m.group("variant") if m else None

def _linestyle_for(label: str) -> str:
    v = _extract_mapl_variant(label)
    if v == "S": return "-"   # MAPL server: solid
    if v == "C": return "--"  # MAPL client: dashed
    return "-"                # others: solid

# --------------------- helpers ---------------------

def _concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["dataset","algorithm_display","measure","seed","client_id",
                 "iteration","accuracy","client_net_type_name","client_net_type_value",
                 "alpha_dich","lambda_ditto","server_data_ratio","_path","algorithm"]
    )

def _load_subfig_df(subfig_dir: Path, inspect: bool) -> pd.DataFrame:
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

def _apply_measure_filter_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Keep client rows (all algs). Optionally keep server rows (MAPL only)."""
    if "measure" not in df.columns:
        return df
    if INCLUDE_SERVER_FOR_MAPL:
        mask = (df["measure"].astype(str) == PLOT_MEASURE_BASE) | \
               ((df["measure"].astype(str) == "server") &
                (df["algorithm_display"].astype(str).str.startswith("MAPL_S-")))
        return df[mask]
    else:
        return df[df["measure"].astype(str) == PLOT_MEASURE_BASE]

def figure_diff_benchmarks(figset_dir: Path, out_root: Path, *, alpha_value: int, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return
    subfigs = [d for d in sorted(figset_dir.iterdir()) if d.is_dir()]
    if not subfigs:
        print(f"[WARN] No subfig folders under {figset_dir}"); return
    loaded = [(sf, _load_subfig_df(sf, inspect)) for sf in subfigs]
    filtered = []
    for sf, df in loaded:
        if df.empty: continue
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
        dfp = df[df["dataset"].astype(str) == dataset_choice]
        dfp = _apply_measure_filter_for_comparison(dfp)
        if dfp.empty:
            ax.set_visible(False); continue
        g = (
            dfp.groupby(["algorithm_display","iteration"], dropna=False)["accuracy"]
               .mean().reset_index().rename(columns={"accuracy":"avg_accuracy"})
               .sort_values(["algorithm_display","iteration"])
        )
        for lab in sorted(g["algorithm_display"].astype(str).unique().tolist()):
            sub_lab = g[g["algorithm_display"].astype(str) == lab]
            ax.plot(sub_lab["iteration"], sub_lab["avg_accuracy"],
                    marker=None, linewidth=2.0, label=str(lab),
                    color=_get_color_for_label(lab),
                    linestyle=_linestyle_for(lab))
        ax.set_title(_right_of_dash(dataset_choice))
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)

    fig.supxlabel("Iteration")
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

    fig_w = max(5.0 * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.5), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    # global y-limits for consistency
    all_grp = []
    for _, df in loaded:
        dfm = _apply_measure_filter_for_comparison(df)
        if dfm.empty: continue
        g = (dfm.groupby(["algorithm_display","iteration"], dropna=False)["accuracy"]
                .mean().reset_index().rename(columns={"accuracy":"avg_accuracy"}))
        all_grp.append(g)
    if not all_grp:
        print(f"[WARN] No rows for comparison measures in {figset_dir}."); return
    big = pd.concat(all_grp, ignore_index=True)
    ymin = float(big["avg_accuracy"].min())
    ymax = float(big["avg_accuracy"].max())

    for ax, (_, df) in zip(axes, loaded):
        dfp = _apply_measure_filter_for_comparison(df)
        if dfp.empty:
            ax.set_visible(False); continue

        g = (
            dfp.groupby(["algorithm_display","iteration"], dropna=False)["accuracy"]
               .mean().reset_index().rename(columns={"accuracy":"avg_accuracy"})
               .sort_values(["algorithm_display","iteration"])
        )
        dom_net = dfp["client_net_type_name"].astype(str).value_counts().index[0] \
                  if "client_net_type_name" in dfp.columns and not dfp.empty else "clients"
        dom_net_disp = TITLE_MAP.get(dom_net, dom_net)

        for lab in sorted(g["algorithm_display"].astype(str).unique().tolist()):
            sub_lab = g[g["algorithm_display"].astype(str) == lab]
            ax.plot(sub_lab["iteration"], sub_lab["avg_accuracy"],
                    marker=None, linewidth=2.0, label=str(lab),
                    color=_get_color_for_label(lab),
                    linestyle=_linestyle_for(lab))
        ax.set_title(f"{dom_net_disp}")
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)
        ax.set_ylim(ymin, ymax)

    fig.supxlabel("Iteration")
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

    # Prefer CIFAR100
    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    # Keep client for all + server for MAPL
    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (client+MAPL server) in {figset_dir}."); return

    if "client_net_type_value" not in df_all.columns:
        df_all["client_net_type_value"] = None
    if "client_net_type_name" not in df_all.columns:
        df_all["client_net_type_name"] = None

    facet_key = df_all["client_net_type_value"].astype(str)
    bad = facet_key.isna() | (facet_key.str.lower().isin(["none", "nan", ""]))
    facet_key = facet_key.mask(bad, df_all["client_net_type_name"].astype(str))

    raw_values = [v for v in facet_key.unique().tolist() if v and v.lower() not in {"none", "nan"}]
    if not raw_values:
        print(f"[WARN] No usable client_net_type_value/name to facet in {figset_dir}."); return

    def _facet_priority(v: str) -> Tuple[int, str]:
        v_low = v.lower()
        if "alex" in v_low:      return (0, v)
        if "resnet" in v_low:    return (1, v)
        if "mobile" in v_low:    return (2, v)
        if "squeeze" in v_low:   return (3, v)
        return (10, v)
    values = sorted(raw_values, key=_facet_priority)[:4]
    n = len(values)

    big = (
        df_all.assign(__facet_key__=facet_key)
              .groupby(["__facet_key__", "algorithm_display", "iteration"], dropna=False)["accuracy"]
              .mean().reset_index().rename(columns={"accuracy": "avg_accuracy"})
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
               .mean().reset_index().rename(columns={"accuracy": "avg_accuracy"})
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
        ax.set_ylabel("Average Accuracy")
        ax.grid(False)
        ax.set_ylim(ymin, ymax)

    fig.supxlabel("Iteration")
    h, l = _legend_from_axes_row(axes)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)), frameon=False, bbox_to_anchor=(0.5, 0.98))

    outfile = out_root / "client_net_type_value.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

# --------------------- Heatmap (Algorithm × Client Net) ---------------------
# Keep this CLIENT-only to avoid mixing measures in the matrix.
def figure_client_net_type_heatmap(figset_dir: Path, out_root: Path, *, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}"); return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    df_all = df_all[df_all["measure"].astype(str) == "client"]
    if df_all.empty:
        print(f"[WARN] No CLIENT rows for heatmap in {figset_dir}."); return

    if "client_net_type_value" not in df_all.columns:
        df_all["client_net_type_value"] = None
    if "client_net_type_name" not in df_all.columns:
        df_all["client_net_type_name"] = None

    facet_value = df_all["client_net_type_value"].astype(str)
    bad = facet_value.isna() | (facet_value.str.lower().isin(["none", "nan", ""]))
    facet_key = facet_value.mask(bad, df_all["client_net_type_name"].astype(str))
    usable = facet_key.astype(str).str.len() > 0
    df_all = df_all[usable]
    if df_all.empty:
        print(f"[WARN] No usable client_net_type_value/name to facet in {figset_dir}."); return

    grp = (
        df_all.assign(__facet__=facet_key)
              .groupby(["__facet__", "algorithm_display", "iteration"], dropna=False)["accuracy"]
              .mean()
              .reset_index()
              .rename(columns={"accuracy": "avg_accuracy"})
    )
    if grp.empty:
        print(f"[WARN] No grouped accuracy rows for heatmap."); return

    last_it = grp.groupby(["__facet__", "algorithm_display"], dropna=False)["iteration"].max().reset_index()
    final = last_it.merge(grp, on=["__facet__", "algorithm_display", "iteration"], how="left")

    def _facet_priority(v: str):
        vl = str(v).lower()
        if "alex" in vl:    return (0, str(v))
        if "resnet" in vl:  return (1, str(v))
        if "mobile" in vl:  return (2, str(v))
        if "squeeze" in vl: return (3, str(v))
        return (10, str(v))
    facets = sorted(final["__facet__"].astype(str).unique().tolist(), key=_facet_priority)

    def _pretty_facet_title(raw: str) -> str:
        return _net_title_from_map(raw, raw)
    col_titles = [_pretty_facet_title(c) for c in facets]

    known_order = {name: i for i, name in enumerate(GLOBAL_ALG_COLOR_ORDER)}
    algs = sorted(final["algorithm_display"].astype(str).unique().tolist(),
                  key=lambda a: (known_order.get(_canon_algo_name(a), 10_000), a))

    mat = np.full((len(algs), len(facets)), np.nan, dtype=float)
    idx_by_alg = {a: i for i, a in enumerate(algs)}
    idx_by_fac = {f: j for j, f in enumerate(facets)}
    for _, row in final.iterrows():
        a = str(row["algorithm_display"])
        f = str(row["__facet__"])
        v = float(row["avg_accuracy"])
        if a in idx_by_alg and f in idx_by_fac:
            mat[idx_by_alg[a], idx_by_fac[f]] = v

    fig_h = max(0.45 * len(algs) + 1.8, 3.8)
    fig_w = max(1.2 * len(facets) + 2.2, 5.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final Avg Accuracy", rotation=90)

    ax.set_xticks(range(len(facets)))
    ax.set_xticklabels(col_titles, rotation=30, ha="right")
    ax.set_yticks(range(len(algs)))
    ax.set_yticklabels(algs)

    for i in range(len(algs)):
        for j in range(len(facets)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title("Final Accuracy by Algorithm × Client Net Type (CLIENT only)")
    ax.set_xlabel("Client Net Type")
    ax.set_ylabel("Algorithm")
    ax.grid(False)

    fig.tight_layout()
    outfile = out_root / "client_net_type_heatmap.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

# --------------------- Global-data-size (ONE plot) ---------------------

def figure_global_data_size_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool, use_server_measure: bool = False):
    """
    Single figure, multiple curves (one per immediate subfolder).
    Forces SERVER measure if use_server_measure=True. Legend labels are numeric ratios.
    """
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    curve_dirs = [d for d in sorted(figset_dir.iterdir()) if d.is_dir()]
    if not curve_dirs:
        print(f"[WARN] No curve folders under {figset_dir}"); return

    curves = []
    for cd in curve_dirs:
        df = load_rows_from_dir(cd, inspect=inspect, alg_hint=None)
        if df.empty:
            continue

        mask_cifar = df["dataset"].astype(str).str.lower() == "cifar100"
        if mask_cifar.any():
            df = df[mask_cifar]
        if df.empty:
            continue

        # Prefer MAPL rows; fall back if none match
        is_mapl = df["algorithm"].astype(str).str.lower().str.contains("mapl") | \
                  df["algorithm_display"].astype(str).str.lower().str.contains("mapl")
        df_mapl = df[is_mapl]
        if not df_mapl.empty:
            df = df_mapl

        target_measure = "server" if use_server_measure else "client"
        if "measure" in df.columns:
            df = df[df["measure"].astype(str) == target_measure]
        if df.empty:
            continue

        if "server_data_ratio" not in df.columns:
            df["server_data_ratio"] = None
        r = df["server_data_ratio"].dropna()
        if len(r) > 0:
            ratio = float(r.iloc[0])
            label = f"{ratio:g}"
        else:
            ratio, _ = find_server_data_ratio({}, fallback_from=cd.name)
            label = f"{ratio:g}" if ratio is not None else cd.name

        g = (
            df.groupby(["iteration"], dropna=False)["accuracy"]
              .mean().reset_index().rename(columns={"accuracy":"avg_accuracy"})
              .sort_values(["iteration"])
        )
        if not g.empty:
            curves.append((label, g))

    if not curves:
        print(f"[WARN] No usable rows with server_data_ratio in {figset_dir}"); return

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
    cmap = plt.get_cmap("tab10")

    def _ratio_key(lbl: str):
        try:
            return float(lbl)
        except Exception:
            return float("inf")

    curves.sort(key=lambda t: _ratio_key(t[0]))

    for i, (lab, g) in enumerate(curves):
        ax.plot(
            g["iteration"], g["avg_accuracy"],
            linewidth=2.0, marker=None,
            label=lab, color=cmap(i % 10)
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Accuracy")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=min(6, len(curves)), frameon=False)

    suffix = "server" if use_server_measure else "client"
    outfile = out_root / f"{figset_dir.name}_by_server_data_ratio_{suffix}.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"), help="Root results folder (contains figure-set folders)")
    ap.add_argument("--figdir", type=Path, default=Path("figures"), help="Where to save PDFs")
    ap.add_argument("--inspect", action="store_true", help="Print detected paths/keys while loading")
    ap.add_argument("--lambda_folder", type=str, default="examine_lambda_for_ditto")
    ap.add_argument("--alpha", type=int, default=5)
    ap.add_argument("--facet_client_net_dir", type=Path, default=Path("results/diff_clients_nets"),
                    help="Folder whose immediate subfolders are algorithm folders; used to facet by client_net_type._value_")
    ap.add_argument("--global_data_size_folder", type=str, default="global_data_size",
                    help="Folder under --root where each immediate subfolder represents one curve/ratio.")

    args = ap.parse_args()

    # Global-data-size: SERVER-only, labels are numeric ratios
    figure_global_data_size_oneplot(
        args.root / args.global_data_size_folder,
        args.figdir,
        inspect=args.inspect,
        use_server_measure=True
    )

    # Comparison figures: client for all algorithms + MAPL server overlay
    figure_diff_clients_nets(args.root / "diff_clients_nets", args.figdir, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "diff_benchmarks_05", args.figdir, alpha_value=args.alpha, inspect=args.inspect)
    figure_by_client_net_type_value(args.root / "same_client_nets", args.figdir, inspect=args.inspect)
    figure_client_net_type_heatmap(args.root / "same_client_nets", args.figdir, inspect=args.inspect)

if __name__ == "__main__":
    main()
