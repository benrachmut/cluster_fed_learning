#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_algorithms_avg_accuracy.py

Adds new figures:
  1) Facets subplots by `client_net_type._value_` (row/grid of subplots).
  2) Heatmap of final accuracy: Algorithm × Client Net Type.
  3) MAPL/global-data-size: ONE plot (one curve per subfolder, labeled by server_data_ratio).
  4) NEW: client_scale: ONE plot (one curve per distinct summary[num_clients], seeds averaged).

Existing figsets:
  - diff_clients_nets/       (grid of subplots by subfolder)
  - diff_benchmarks_05/      (row of subplots by subfolder; alpha filter)
  - examine_lambda_for_ditto (single figure; curves by λ_ditto)

Outputs (PDF + CSV):
  - figures/same_client_nets/client_net_type_value.pdf / .csv
  - figures/same_client_nets/client_net_type_heatmap.pdf / .csv
  - figures/<global_data_size_folder>/<global_data_size_folder>_by_server_data_ratio_{client|server}.pdf / .csv
  - figures/diff_clients_nets/diff_clients_nets.pdf / .csv
  - figures/diff_benchmarks_05/diff_benchmarks_05.pdf / .csv
  - figures/client_scale/client_scale.pdf
  - figures/client_scale/client_scale_iter.csv
  - figures/client_scale/client_scale_final.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# =====================================================================
# Global style: fonts + legend sizes
# =====================================================================
BASE_FONT_SIZE = 16
TITLE_FONT_SIZE = 16
LEGEND_FONT_SIZE = 15

plt.rcParams.update({
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": BASE_FONT_SIZE,
    "xtick.labelsize": BASE_FONT_SIZE - 2,
    "ytick.labelsize": BASE_FONT_SIZE - 2,
    "legend.fontsize": LEGEND_FONT_SIZE,
})

# ---------------------------------------------------------------------
# Presentation policy:
# - Non-MAPL algorithms -> plot CLIENT curves.
# - MAPL -> show SERVER-only curves; label as "MAPL" (drop _S/-x suffixes).
# ---------------------------------------------------------------------
PLOT_MEASURE_BASE = "client"
INCLUDE_SERVER_FOR_MAPL = True

TITLE_MAP = {
    "rndNet":    "{AlexNet, ResNet, MobileNet, SqueezeNet}",
    "rndStrong": "{AlexNet, ResNet}",
    "rndWeak":   "{MobileNet, SqueezeNet}",
    "AlexMobile":"{AlexNet, MobileNet}",
    "AlexSqueeze":"{AlexNet, SqueezeNet}",
    "ResMobile":"{ResNet, MobileNet}",
    "ResNetSqueeze":"{ResNet, SqueezeNet}",
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
    "MAPL",      # normalized name for MAPL (server-only in plots)
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
            "lambda_consistency",          # <<< NEW
            "server_data_ratio",
            "num_clients",
            "server_input_tech_name",
            "distill_temperature",
            "_path",
            "algorithm",
        ]
    )

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

# ---- NEW: dataset-aware accuracy map selectors (Top-1 vs Top-5) ----

_IMAGENET_TOP5_DATASETS = {"imagenetr", "tinyimagenet", "tiny-imageNet", "tiny_imageNet"}

def _is_top5_dataset(ds: Optional[str]) -> bool:
    if not ds: return False
    dsl = str(ds).lower()
    return ("imagenetr" in dsl) or ("tinyimagenet" in dsl)

def find_client_accuracy_map_topk(root: Json, k: int):
    # try exact key first
    v, p = find_exact_key_map(root, f"client_accuracy_per_client_{k}")
    if v is not None: return v, p
    # light fallback: accept any client-like map
    c: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if looks_like_client_accuracy_map(v) and any("client" in seg.lower() for seg in p):
            c.append((p, v))
    if c:
        def score(pv):
            p,_ = pv; s=0
            if any("summary" in seg.lower() for seg in p): s+=2
            if any("client" in seg.lower() for seg in p): s+=1
            # prefer explicit *_5 over *_1 if present in path text
            if any("5" in seg for seg in p): s+=1
            return s
        c.sort(key=score, reverse=True)
        return c[0][1], c[0][0]
    return None, None

def find_server_accuracy_map_topk(root: Json, k: int):
    # common naming: server_accuracy_per_client_{k}_max OR server_accuracy_per_client_{k}
    for key in (f"server_accuracy_per_client_{k}_max", f"server_accuracy_per_client_{k}"):
        v, p = find_exact_key_map(root, key)
        if v is not None: return v, p
    # fallback: any server-like map
    c: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if looks_like_client_accuracy_map(v) and any("server" in seg.lower() for seg in p):
            c.append((p, v))
    if c:
        def score(pv):
            p,_=pv; s=0
            if any("summary" in seg.lower() for seg in p): s+=2
            if any("server" in seg.lower() for seg in p): s+=1
            if any(str(k) in seg for seg in p): s+=1
            return s
        c.sort(key=score, reverse=True)
        return c[0][1], c[0][0]
    return None, None

def find_client_accuracy_map_anywhere(root: Json):
    # kept for compatibility (Top-1 default)
    return find_client_accuracy_map_topk(root, 1)

def find_server_accuracy_map(root: Json):
    # kept for compatibility (Top-1 default)
    return find_server_accuracy_map_topk(root, 1)

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
                if v > 1.0: val = v / 100.0
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

def find_num_clients(root: Json):
    """Extract summary['num_clients'] as int if present."""
    try:
        if isinstance(root, dict) and "summary" in root:
            s = root["summary"]
            if isinstance(s, dict) and "num_clients" in s:
                v = s["num_clients"]
                for caster in (int, lambda x: int(float(x))):
                    try:
                        return caster(v), ("summary","num_clients")
                    except Exception:
                        pass
    except Exception:
        pass
    for p, v in walk(root):
        if p and p[-1] == "num_clients":
            for caster in (int, lambda x: int(float(x))):
                try:
                    return caster(v), p
                except Exception:
                    pass
            return None, p
    return None, None

# --------------------- loading ---------------------
def find_lambda_consistency(root: Json):
    """
    Extract summary['lambda_consistency'] as float if present.
    """
    try:
        if isinstance(root, dict) and "summary" in root:
            s = root["summary"]
            if isinstance(s, dict) and "lambda_consistency" in s:
                v = s["lambda_consistency"]
                try:
                    return float(v), ("summary", "lambda_consistency")
                except Exception:
                    try:
                        return float(str(v).strip()), ("summary", "lambda_consistency")
                    except Exception:
                        return None, ("summary", "lambda_consistency")
    except Exception:
        pass

    for p, v in walk(root):
        if p and p[-1] == "lambda_consistency":
            try:
                return float(v), p
            except Exception:
                try:
                    return float(str(v).strip()), p
                except Exception:
                    return None, p
    return None, None

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

        # ----------- Top-1 vs Top-5 selection per dataset -----------
        use_top5 = _is_top5_dataset(ds_name)
        topk = 5 if use_top5 else 1

        client_map, _        = find_client_accuracy_map_topk(raw, topk)
        server_map, _        = find_server_accuracy_map_topk(raw, topk)
        # ------------------------------------------------------------

        server_net_val, _    = find_server_net_type_value(raw)
        client_net_name, _   = find_client_net_type_name(raw)
        client_net_value, _  = find_client_net_type_value(raw)
        alpha_dich, _        = find_alpha_dich(raw)
        lambda_ditto, _      = find_lambda_ditto(raw)
        server_data_ratio, _ = find_server_data_ratio(raw, fallback_from=p)
        num_clients, _       = find_num_clients(raw)
        lambda_consistency, _ = find_lambda_consistency(raw)

        # ---- NEW: grab server_input_tech and distill_temperature from summary ----
        server_input_tech_name = None
        distill_temperature = None
        try:
            s = raw.get("summary", {}) if isinstance(raw, dict) else {}
            if isinstance(s, dict):
                sit = s.get("server_input_tech", None)
                if isinstance(sit, dict):
                    server_input_tech_name = sit.get("_name_") or sit.get("_value_")
                elif isinstance(sit, (str, int, float)):
                    server_input_tech_name = str(sit)

                if "distill_temperature" in s and s["distill_temperature"] is not None:
                    try:
                        distill_temperature = float(s["distill_temperature"])
                    except Exception:
                        distill_temperature = None
        except Exception:
            pass
        # -------------------------------------------------------------------------

        is_mapl = bool(alg_name and "mapl" in str(alg_name).lower())
        used_any = False

        # client measure (all algs) -- MAPL_C is filtered out later
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
                        try:
                            iteration = int(float(iter_k))
                        except Exception:
                            continue
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
                        "num_clients": num_clients,
                        "server_input_tech_name": server_input_tech_name,
                        "distill_temperature": distill_temperature,
                        "lambda_consistency": lambda_consistency,  # <<< NEW

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
                        try:
                            iteration = int(float(iter_k))
                        except Exception:
                            continue
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
                        "num_clients": num_clients,
                        "server_input_tech_name": server_input_tech_name,
                        "distill_temperature": distill_temperature,
                        "lambda_consistency": lambda_consistency,  # <<< NEW

                    })
                    used_any = True

        if used_any:
            files_used += 1

    if inspect:
        print(f"[SUMMARY] scanned={files_scanned}, used={files_used} in {dir_path}")

    cols = [
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
        "lambda_consistency",        # <<< ADD THIS LINE
        "server_data_ratio",
        "num_clients",
        "server_input_tech_name",
        "distill_temperature",
        "_path",
        "algorithm",
    ]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# --------------------- MAPL normalization + linestyle ---------------------

_mapl_label_re = re.compile(r"^MAPL_(?P<variant>[CS])-(?P<x>.+)$")
def _extract_mapl_variant(label: str) -> Optional[str]:
    m = _mapl_label_re.match(label); return m.group("variant") if m else None

def _linestyle_for(label: str) -> str:
    v = _extract_mapl_variant(label)
    if v == "S": return "-"   # MAPL server: solid
    if v == "C": return "--"  # MAPL client: dashed
    return "-"                # others: solid

def _normalize_mapl_presentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop MAPL client rows (MAPL_C-*) and rename MAPL_S-* -> MAPL.
    """
    if df is None or df.empty or "algorithm_display" not in df.columns:
        return df
    df = df.copy()
    algdisp = df["algorithm_display"].astype(str)

    # Drop client-side MAPL (MAPL_C-*)
    mask_c = algdisp.str.startswith("MAPL_C-")
    if mask_c.any():
        df = df[~mask_c]

    # Rename server-side MAPL (MAPL_S-*) to "MAPL"
    if "algorithm_display" in df.columns:
        mask_s = df["algorithm_display"].astype(str).str.startswith("MAPL_S-")
        if mask_s.any():
            df.loc[mask_s, "algorithm_display"] = "MAPL"

    return df

# --------------------- CSV stats helpers ---------------------

def _final_stats_for_panel(df_panel: pd.DataFrame, *, subfigure: str) -> pd.DataFrame:
    """
    For each algorithm in this panel/subfigure:
      - find its maximum iteration present in df_panel
      - compute mean/std/sem/n at that iteration
    Assumes df_panel already filtered & normalized (MAPL policy) and for the target dataset/facet.
    """
    if df_panel is None or df_panel.empty:
        return pd.DataFrame(columns=["subfigure","algorithm","max_iteration","mean","std","sem","n"])
    out_rows = []
    for alg, dfa in df_panel.groupby("algorithm_display"):
        max_it = pd.to_numeric(dfa["iteration"], errors="coerce").dropna().max()
        if pd.isna(max_it):
            continue
        dfit = dfa[dfa["iteration"] == max_it].copy()
        acc = pd.to_numeric(dfit["accuracy"], errors="coerce").dropna()
        if acc.empty:
            continue
        n = int(acc.shape[0])
        mean = float(acc.mean())
        std = float(acc.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        out_rows.append({
            "subfigure": subfigure,
            "algorithm": str(alg),
            "max_iteration": int(max_it),
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
        })
    return pd.DataFrame(out_rows)

def _save_stats_csv(rows: List[pd.DataFrame], outfile_pdf: Path):
    df = _concat_or_empty(rows)
    csv_path = outfile_pdf.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        # write header-only file for predictability
        df = pd.DataFrame(columns=["subfigure","algorithm","max_iteration","mean","std","sem","n"])
    df.to_csv(_win_long_abs(csv_path), index=False)
    print(f"[OK] Saved CSV: {csv_path}")

# --------------------- helpers ---------------------

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
    """
    Enforce presentation policy:
      - Non-MAPL algorithms: keep CLIENT rows.
      - MAPL: keep SERVER rows only.
      - Rename any MAPL_S-* -> MAPL (legend/label clean), drop MAPL_C-* entirely.
    """
    if df is None or df.empty or "measure" not in df.columns or "algorithm_display" not in df.columns:
        return df
    df = df.copy()
    algdisp = df["algorithm_display"].astype(str)
    is_mapl = algdisp.str.contains(r"^MAPL", regex=True)
    keep_non_mapl = (~is_mapl) & (df["measure"].astype(str) == PLOT_MEASURE_BASE)
    keep_mapl     = (is_mapl) & (df["measure"].astype(str) == "server")
    out = df[keep_non_mapl | keep_mapl]
    return _normalize_mapl_presentation(out)

def figure_diff_benchmarks(figset_dir: Path, out_root: Path, *, alpha_value: int, inspect: bool):
    """
    NOTE: For ImageNet-R and TinyImageNet, the loader already pulled Top-5 accuracy;
          for other datasets it pulled Top-1. This function just plots what's loaded.
    """
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return
    subfigs = [d for d in sorted(figset_dir.iterdir()) if d.is_dir()]
    if not subfigs:
        print(f"[WARN] No subfig folders under {figset_dir}"); return
    loaded = [(sf, _load_subfig_df(sf, inspect)) for sf in subfigs]

    stats_rows: List[pd.DataFrame] = []
    filtered = []
    for sf, df in loaded:
        if df.empty: continue
        sub = df[(df["alpha_dich"] == alpha_value)]
        if not sub.empty:
            filtered.append((sf, sub))
    if not filtered:
        print(f"[WARN] No rows for alpha={alpha_value} in {figset_dir}."); return

    n = len(filtered)
    fig_w = max(5.0 * n, 5.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.5), sharex=False, sharey=False, squeeze=False)
    axes = axes.flatten()

    for ax, (sf, df) in zip(axes, filtered):
        # choose a dataset to show from this subfolder. If ImageNetR/TinyImageNet exist, prefer them.
        ds_vals = df["dataset"].astype(str)
        choice = None
        for candidate in ["ImageNetR", "TinyImageNet", "CIFAR100"]:
            if candidate in ds_vals.values:
                choice = candidate; break
        if choice is None:
            choice = ds_vals.value_counts().index[0]

        dfp = df[df["dataset"].astype(str) == choice]
        dfp = _apply_measure_filter_for_comparison(dfp)
        if dfp.empty:
            ax.set_visible(False); continue

        # CSV stats for this subfigure
        subfigure_label = _right_of_dash(choice)
        stats_rows.append(_final_stats_for_panel(dfp, subfigure=subfigure_label))

        g = (
            dfp.groupby(["algorithm_display","iteration"], dropna=False)["accuracy"]
               .mean().reset_index().rename(columns={"accuracy":"avg_accuracy"})
               .sort_values(["algorithm_display","iteration"])
        )
        for lab in sorted(g["algorithm_display"].astype(str).unique().tolist()):
            sub_lab = g[g["algorithm_display"].astype(str) == lab]
            ax.plot(sub_lab["iteration"], sub_lab["avg_accuracy"],
                    marker=None, linewidth=3.0, label=str(lab),
                    color=_get_color_for_label(lab),
                    linestyle=_linestyle_for(lab))
        ax.set_title(subfigure_label)

        # y-axis label reflects metric choice (Top-5 for ImageNetR/TinyImageNet, else Top-1)
        if _is_top5_dataset(choice):
            ax.set_ylabel("Top-5 Accuracy")
        else:
            ax.set_ylabel("Top-1 Accuracy")

        ax.grid(False)

    fig.supxlabel("Iteration", fontsize=BASE_FONT_SIZE)
    h, l = _legend_from_axes_row(axes)
    # bring legend closer: more space for subplots, legend slightly down
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)),
                   frameon=False, bbox_to_anchor=(0.5, 0.96),
                   fontsize=LEGEND_FONT_SIZE)

    outfile = out_root / f"{figset_dir.name}.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")
    _save_stats_csv(stats_rows, outfile)

def _load_subfig_df(subfig_dir: Path, inspect: bool) -> pd.DataFrame:
    alg_dirs = [a for a in sorted(subfig_dir.iterdir()) if a.is_dir()]
    frames = []
    for alg_dir in alg_dirs:
        frames.append(load_rows_from_dir(alg_dir, inspect=inspect, alg_hint=alg_dir.name))
    return _concat_or_empty(frames)

def figure_diff_clients_nets(figset_dir: Path, out_root: Path, *, inspect: bool, max_cols: int = 4):
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

    n = len(loaded)
    ncols = max(1, min(max_cols, n))
    nrows = math.ceil(n / ncols)

    # scale figure size by grid
    fig_w = max(5.0 * ncols, 5.0)
    fig_h = max(4.5 * nrows, 4.5)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True,  # shared axes
        squeeze=False
    )
    axes = axes.flatten()

    # global y-limits for consistency
    all_grp = []
    for _, df in loaded:
        dfm = _apply_measure_filter_for_comparison(df)
        if dfm.empty:
            continue
        g = (
            dfm.groupby(["algorithm_display", "iteration"], dropna=False)["accuracy"]
               .mean()
               .reset_index()
               .rename(columns={"accuracy": "avg_accuracy"})
        )
        all_grp.append(g)
    if not all_grp:
        print(f"[WARN] No rows for comparison measures in {figset_dir}."); return
    big = pd.concat(all_grp, ignore_index=True)
    ymin = float(big["avg_accuracy"].min())
    ymax = float(big["avg_accuracy"].max())

    stats_rows: List[pd.DataFrame] = []

    # plot each subfigure
    for ax, (sf, df) in zip(axes, loaded):
        dfp = _apply_measure_filter_for_comparison(df)
        if dfp.empty:
            ax.set_visible(False)
            continue

        dom_net = (
            dfp["client_net_type_name"].astype(str).value_counts().index[0]
            if "client_net_type_name" in dfp.columns and not dfp.empty
            else "clients"
        )
        dom_net_disp = TITLE_MAP.get(dom_net, dom_net)

        # CSV stats for this panel
        stats_rows.append(_final_stats_for_panel(dfp, subfigure=dom_net_disp))

        g = (
            dfp.groupby(["algorithm_display", "iteration"], dropna=False)["accuracy"]
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
                linewidth=3.0,
                label=str(lab),
                color=_get_color_for_label(lab),
                linestyle=_linestyle_for(lab),
            )

        ax.set_title(f"{dom_net_disp}")
        ax.grid(False)
        ax.set_ylim(ymin, ymax)

        # ensure y tick labels are visible on *all* subplots
        ax.tick_params(labelleft=True)

    # hide any trailing empty axes if grid is larger than loaded
    for ax in axes[n:]:
        ax.set_visible(False)

    # Axis titles once for the whole figure
    fig.supxlabel("Iteration", fontsize=BASE_FONT_SIZE)         # X title (already once)
    fig.supylabel("Top-1 Accuracy", fontsize=BASE_FONT_SIZE)  # Y title once

    # legend
    first_ax = next((a for a in axes[:n] if a.get_visible()), None)
    h, l = (first_ax.get_legend_handles_labels() if first_ax else ([], []))

    # bring legend closer
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    if h:
        fig.legend(
            h, l,
            loc="upper center",
            ncol=min(6, len(h)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
            fontsize=LEGEND_FONT_SIZE,
        )

    outfile = out_root / f"{figset_dir.name}.pdf"
    _savefig_longpath(fig, outfile)
    plt.close(fig)
    print(f"[OK] Saved: {outfile}")
    _save_stats_csv(stats_rows, outfile)

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

    # Keep client for non-MAPL + server for MAPL (normalized)
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

    stats_rows: List[pd.DataFrame] = []

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

        # CSV stats for this facet
        stats_rows.append(_final_stats_for_panel(sub, subfigure=title_name))

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
                linewidth=3.0,
                label=str(lab),
                color=_get_color_for_label(lab),
                linestyle=_linestyle_for(lab),
            )

        ax.set_title(title_name)
        # <- NO per-axis y-label here anymore
        ax.grid(False)
        ax.set_ylim(ymin, ymax)

    # Single global axis labels
    fig.supxlabel("Iteration", fontsize=BASE_FONT_SIZE)
    fig.supylabel("Top-1 Accuracy", fontsize=BASE_FONT_SIZE)  # <<< y-title once

    h, l = _legend_from_axes_row(axes)

    # Legend: force a single row (all entries in one row)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if h:
        fig.legend(
            h, l,
            loc="upper center",
            ncol=len(h),          # single row
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
            fontsize=LEGEND_FONT_SIZE,
        )

    outfile = out_root / "client_net_type_value.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")
    _save_stats_csv(stats_rows, outfile)

# --------------------- Heatmap (Algorithm × Client Net) ---------------------
# (intentionally omitted here; unchanged in your version)

# --------------------- Global-data-size (ONE plot) ---------------------

def figure_global_data_size_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool, use_server_measure: bool = False):
    """
    Single figure, multiple curves (one per immediate subfolder).
    Forces SERVER measure if use_server_measure=True. Legend labels are numeric ratios.
    Also emits a CSV with final stats per ratio/curve.
    """
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    curve_dirs = [d for d in sorted(figset_dir.iterdir()) if d.is_dir()]
    if not curve_dirs:
        print(f"[WARN] No curve folders under {figset_dir}"); return

    curves = []
    stats_rows: List[pd.DataFrame] = []
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

        # CSV stats for this ratio curve
        stats_rows.append(_final_stats_for_panel(df, subfigure=str(label)))

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
            linewidth=3.0, marker=None,
            label=lab, color=cmap(i % 10)
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)

    # Figure-level legend outside (top center), consistent with other figs
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if h:
        fig.legend(
            h, l,
            loc="upper center",
            ncol=min(6, len(curves)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
            fontsize=LEGEND_FONT_SIZE,
        )

    suffix = "server" if use_server_measure else "client"
    outfile = out_root / f"{figset_dir.name}_by_server_data_ratio_{suffix}.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")
    _save_stats_csv(stats_rows, outfile)

# --------------------- NEW: client_scale (ONE plot; one curve per num_clients) ---------------------

def _maybe_scale_to_percent(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    try:
        mx = np.nanmax(arr)
        if mx <= 1.05:
            return arr * 100.0
    except Exception:
        pass
    return arr

def _aggregate_run_mean_over_clients(df_run: pd.DataFrame) -> pd.Series:
    """
    Given rows from a single JSON (same path), average accuracy across clients per iteration.
    Assumes df has columns: iteration, client_id, accuracy.
    Returns a Series indexed by iteration.
    """
    g = df_run.groupby("iteration", dropna=False)["accuracy"].mean().sort_index()
    return g

def _final_stats_for_client_scale(df_iter: pd.DataFrame) -> pd.DataFrame:
    """
    df_iter columns: iter, num_clients, mean, sem, n_seeds
    Return final-iteration stats per num_clients.
    """
    out = []
    for nc, d in df_iter.groupby("num_clients"):
        if d.empty: continue
        max_it = pd.to_numeric(d["iter"], errors="coerce").dropna().max()
        row = d[d["iter"] == max_it]
        if row.empty: continue
        r = row.iloc[0]
        out.append({
            "num_clients": int(nc) if pd.notna(nc) else None,
            "max_iteration": int(max_it),
            "mean": float(r["mean"]),
            "sem": float(r["sem"]),
            "n_seeds": int(r["n_seeds"]),
        })
    return pd.DataFrame(out, columns=["num_clients","max_iteration","mean","sem","n_seeds"])

def figure_client_scale_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool):
    """
    Scans results/client_scale for JSONs of one algorithm/config run across different seeds.
    Each JSON must contain summary['num_clients'].
    We build one curve per num_clients (averaging across seeds).
    Also emits two CSVs: per-iteration stats and final-iteration stats.
    """
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}"); return

    # Prefer CIFAR100 when available
    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    # Apply presentation policy
    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {figset_dir}"); return

    # Require num_clients
    if "num_clients" not in df_all.columns or df_all["num_clients"].isna().all():
        print(f"[WARN] No summary[num_clients] detected in {figset_dir}"); return

    # Ensure types
    df_all = df_all.copy()
    df_all["num_clients"] = pd.to_numeric(df_all["num_clients"], errors="coerce")
    df_all = df_all.dropna(subset=["num_clients"])

    # First: split per file path (run), average over clients per iteration
    iter_rows = []
    for (path, nc, seed), d in df_all.groupby(["_path","num_clients","seed"], dropna=False):
        if d.empty:
            continue
        sub = d[["iteration","client_id","accuracy"]].copy()
        s = _aggregate_run_mean_over_clients(sub)  # Series index=iteration
        if s.empty:
            continue
        iter_rows.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "value": s.values.astype(float),
            "num_clients": nc,
            "seed": seed,
            "_path": path
        }))

    if not iter_rows:
        print(f"[WARN] No per-run curves produced for {figset_dir}"); return

    df_runs = pd.concat(iter_rows, ignore_index=True)

    # If values look like probs, scale to % (per num_clients to be safe)
    def _maybe_scale_group(g: pd.DataFrame) -> pd.DataFrame:
        vals = g["value"].to_numpy(dtype=float)
        g = g.copy()
        g["value"] = _maybe_scale_to_percent(vals)
        return g
    df_runs = df_runs.groupby("num_clients", group_keys=False).apply(_maybe_scale_group)

    # Align lengths across seeds per num_clients by truncating to common min length
    iter_stats = []
    for nc, g in df_runs.groupby("num_clients"):
        mats = []
        for sd, d in g.groupby("seed"):
            d = d.sort_values("iter")
            mats.append(d["value"].to_numpy(dtype=float))
        if not mats:
            continue
        T = min(len(a) for a in mats)
        if T == 0:
            continue
        arr = np.vstack([a[:T] for a in mats])  # shape [S, T]
        iters = np.sort(g["iter"].unique())[:T]
        means = np.nanmean(arr, axis=0)
        sems = (np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])) if arr.shape[0] > 1 else np.zeros(T)
        iter_stats.append(pd.DataFrame({
            "iter": iters.astype(int),
            "num_clients": int(nc),
            "mean": means.astype(float),
            "sem": sems.astype(float),
            "n_seeds": int(arr.shape[0]),
        }))

    if not iter_stats:
        print(f"[WARN] No per-iteration stats available for {figset_dir}"); return

    df_iter = pd.concat(iter_stats, ignore_index=True).sort_values(["num_clients","iter"])

    # Save per-iteration CSV
    out_dir = out_root / "client_scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    iter_csv = out_dir / "client_scale_iter.csv"
    df_iter.to_csv(_win_long_abs(iter_csv), index=False)
    print(f"[OK] Saved: {iter_csv}")

    # Final-iteration CSV
    df_final = _final_stats_for_client_scale(df_iter)
    final_csv = out_dir / "client_scale_final.csv"
    df_final.to_csv(_win_long_abs(final_csv), index=False)
    print(f"[OK] Saved: {final_csv}")

    # Plot (no shading, legend outside like other figs)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))

    # nice numeric order
    for nc in sorted(df_iter["num_clients"].unique().tolist()):
        d = df_iter[df_iter["num_clients"] == nc].sort_values("iter")
        x = d["iter"].to_numpy()
        y = d["mean"].to_numpy(dtype=float)
        ax.plot(x, y, linewidth=3.0, label=f"{int(nc)} clients")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)

    # Global legend outside (top center), single row-style like others
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if h:
        fig.legend(
            h, l,
            loc="upper center",
            ncol=min(6, len(h)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
            fontsize=LEGEND_FONT_SIZE,
        )

    out_pdf = out_dir / "client_scale.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

# --------------------- CLI ---------------------

def _extract_tech_temp_from_path(path_str: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Given a JSON path (string), open it, look in summary for:
      - summary['server_input_tech']['_name_'] (or '_value_' / str(server_input_tech))
      - summary['distill_temperature']
    Returns (tech_name, temperature) or (None, None) if not found.
    """
    p = Path(path_str)
    try:
        with _open_json_win_safe(p) as fh:
            raw = json.load(fh)
    except Exception as e:
        print(f"[WARN] failed to read {p}: {e}")
        return None, None

    tech = None
    temp = None

    # Prefer direct summary access
    try:
        if isinstance(raw, dict) and "summary" in raw:
            s = raw["summary"]
            if isinstance(s, dict):
                sit = s.get("server_input_tech", None)
                if isinstance(sit, dict):
                    tech = sit.get("_name_") or sit.get("_value_")
                elif isinstance(sit, (str, int, float)):
                    tech = str(sit)

                if "distill_temperature" in s and s["distill_temperature"] is not None:
                    try:
                        temp = float(s["distill_temperature"])
                    except Exception:
                        temp = None
    except Exception:
        pass

    # Fallback: scan the JSON (if missing above)
    if tech is None or temp is None:
        for path, val in walk(raw):
            if tech is None:
                # anything with 'server_input_tech' in the path
                pl = [seg.lower() for seg in path]
                if any("server_input_tech" in seg for seg in pl) and isinstance(val, (str, int, float)):
                    tech = str(val)
            if temp is None and path and path[-1] == "distill_temperature":
                try:
                    temp = float(val)
                except Exception:
                    temp = None
            if tech is not None and temp is not None:
                break

    return tech, temp

def figure_temp_serverinput_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool):
    """
    results/temp_serverinput:
      - all runs are MAPL.
      - One curve per (server_input_tech_name, distill_temperature) combination.

    For each combo:
      - For each run (JSON file / seed), average accuracy across clients per iteration.
      - Then average across seeds, truncating all seeds to the common min length
        (same style as figure_client_scale_oneplot).

    X-axis: iteration
    Y-axis: Top-1 Accuracy (or Top-5, depending on dataset; loader already chose).
    """
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)")
        return

    # Load all JSONs under this folder (including possible subdirs)
    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}")
        return

    # Prefer CIFAR100 if present
    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    if df_all.empty:
        print(f"[WARN] No rows left after CIFAR100 filtering in {figset_dir}")
        return

    # Apply standard presentation policy: MAPL -> server only, others -> client
    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {figset_dir}")
        return

    # We MUST have server_input_tech_name and distill_temperature
    if "server_input_tech_name" not in df_all.columns or "distill_temperature" not in df_all.columns:
        print(f"[WARN] server_input_tech_name/distill_temperature missing in df under {figset_dir}")
        return

    df_all = df_all.copy()
    df_all["server_input_tech_name"] = df_all["server_input_tech_name"].astype(str)
    df_all["distill_temperature"] = pd.to_numeric(df_all["distill_temperature"], errors="coerce")

    # Drop rows with missing metadata
    df_all = df_all.dropna(subset=["distill_temperature"])
    df_all = df_all[df_all["server_input_tech_name"].str.lower().ne("none")]
    if df_all.empty:
        print(f"[WARN] No rows with valid server_input_tech_name and distill_temperature in {figset_dir}")
        return

    if "_path" not in df_all.columns:
        print(f"[WARN] _path column missing in df under {figset_dir}")
        return

    # -------- Step 1: per-run curves (avg over clients) --------
    run_rows = []
    for (tech, temp, path, seed), d in df_all.groupby(
        ["server_input_tech_name", "distill_temperature", "_path", "seed"],
        dropna=False,
    ):
        if d.empty:
            continue
        sub = d[["iteration", "client_id", "accuracy"]].copy()
        s = _aggregate_run_mean_over_clients(sub)  # Series: index=iteration, values=accuracy
        if s.empty:
            continue
        run_rows.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "value": s.values.astype(float),
            "server_input_tech_name": tech,
            "distill_temperature": float(temp),
            "seed": seed,
            "_path": path,
        }))

    if not run_rows:
        print(f"[WARN] No per-run curves produced for {figset_dir}")
        return

    df_runs = pd.concat(run_rows, ignore_index=True)

    # If values look like probabilities (<=1), scale to %, per (tech,temp) combo
    def _maybe_scale_group_temp(g: pd.DataFrame) -> pd.DataFrame:
        vals = g["value"].to_numpy(dtype=float)
        g = g.copy()
        g["value"] = _maybe_scale_to_percent(vals)
        return g

    df_runs = df_runs.groupby(
        ["server_input_tech_name", "distill_temperature"],
        group_keys=False,
    ).apply(_maybe_scale_group_temp)

    # -------- Step 2: aggregate across seeds per (tech, temp) --------
    iter_stats = []
    for (tech, temp), g in df_runs.groupby(["server_input_tech_name", "distill_temperature"]):
        mats = []
        for seed, d in g.groupby("seed"):
            d = d.sort_values("iter")
            vals = d["value"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            mats.append(vals)

        if not mats:
            continue

        # Truncate to common min length (same as client_scale)
        T = min(len(a) for a in mats)
        if T == 0:
            continue
        arr = np.vstack([a[:T] for a in mats])  # [S, T]

        iters = np.sort(g["iter"].unique())[:T]
        means = np.nanmean(arr, axis=0)
        sems = (np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])) if arr.shape[0] > 1 else np.zeros(T)

        iter_stats.append(pd.DataFrame({
            "iter": iters.astype(int),
            "server_input_tech_name": tech,
            "distill_temperature": float(temp),
            "mean": means.astype(float),
            "sem": sems.astype(float),
            "n_seeds": int(arr.shape[0]),
        }))

    if not iter_stats:
        print(f"[WARN] No per-iteration stats available for {figset_dir}")
        return

    df_iter = pd.concat(iter_stats, ignore_index=True).sort_values(
        ["server_input_tech_name", "distill_temperature", "iter"]
    )

    # -------- Step 3: plot accuracy vs iteration --------
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))

    # Y starts at chosen bounds
    ymin = 15  # or float(df_iter["mean"].min())
    ymax = 50  # or float(df_iter["mean"].max())
    ax.set_ylim(ymin, ymax)

    # X goes only up to iter = 9
    xmin = int(df_iter["iter"].min())
    ax.set_xlim(xmin, 9)

    cmap = plt.get_cmap("tab10")

    # Color per temperature, linestyle per server_input_tech
    temps = sorted(df_iter["distill_temperature"].dropna().unique().tolist())
    temp_color: Dict[float, Any] = {}
    for i, t in enumerate(temps):
        temp_color[t] = cmap(i % 10)

    def _ls_for_tech(tech: str) -> str:
        tl = (tech or "").strip().lower()
        if "mean" in tl:
            return ":"        # dotted  (mean)
        if "max" in tl:
            return "-"        # solid   (max)
        if "median" in tl or "med" in tl:
            return "--"       # dashed  (median)
        return "-"            # default solid

    def _tech_order(tech: str) -> int:
        tl = (tech or "").strip().lower()
        if "max" in tl:
            return 0
        if "mean" in tl:
            return 1
        if "median" in tl or "med" in tl:
            return 2
        return 3  # anything else last

    # Stable order: group by temperature, then by tech type (max, mean, median)
    raw_keys = df_iter[["server_input_tech_name", "distill_temperature"]].drop_duplicates()
    curve_keys = sorted(
        [(row["server_input_tech_name"], float(row["distill_temperature"])) for _, row in raw_keys.iterrows()],
        key=lambda x: (x[1], _tech_order(x[0]))
    )

    for tech, temp in curve_keys:
        d = df_iter[
            (df_iter["server_input_tech_name"] == tech) &
            (df_iter["distill_temperature"] == temp)
        ].sort_values("iter")

        x = d["iter"].to_numpy()
        y = d["mean"].to_numpy(dtype=float)

        color = temp_color.get(temp, cmap(0))
        ls = _ls_for_tech(tech)

        # Legend label: same color for same T; different linestyle per tech
        label = f"T={temp:g}, {tech}"
        ax.plot(
            x,
            y,
            linewidth=3.0,
            marker=None,
            label=label,
            color=color,
            linestyle=ls,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)

    # Legend on the right, 1 column, tight spacing
    h, l = ax.get_legend_handles_labels()

    # Leave only a small margin on the right
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

    if h:
        fig.legend(
            h, l,
            loc="center left",  # right side of plot
            bbox_to_anchor=(0.96, 0.5),  # very close to axes
            ncol=1,  # SINGLE COLUMN
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            borderaxespad=0.3,  # minimal padding between axes & legend
        )

    out_dir = out_root / "temp_serverinput"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "temp_serverinput.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")


def _final_stats_for_mapl_lambda(df_iter: pd.DataFrame) -> pd.DataFrame:
    """
    df_iter columns: iter, lambda_consistency, mean, sem, n_seeds
    Return final-iteration stats per lambda_consistency.
    """
    out = []
    for lam, d in df_iter.groupby("lambda_consistency"):
        if d.empty:
            continue
        max_it = pd.to_numeric(d["iter"], errors="coerce").dropna().max()
        row = d[d["iter"] == max_it]
        if row.empty:
            continue
        r = row.iloc[0]
        out.append({
            "lambda_consistency": float(lam) if pd.notna(lam) else None,
            "max_iteration": int(max_it),
            "mean": float(r["mean"]),
            "sem": float(r["sem"]),
            "n_seeds": int(r["n_seeds"]),
        })
    return pd.DataFrame(out, columns=["lambda_consistency","max_iteration","mean","sem","n_seeds"])



def figure_mapl_lambda_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool):
    """
    results/mapl_lambda:
      - runs are MAPL with different summary['lambda_consistency'] and seeds.
      - One curve per distinct lambda_consistency value (averaged over seeds).

    For each lambda:
      - For each run (JSON file / seed), average accuracy across clients per iteration.
      - Then average across seeds, truncating all seeds to the common min length.

    X-axis: iteration
    Y-axis: Top-1 Accuracy (or % if values were in [0,1]).
    """
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)")
        return

    # Load all JSONs under this folder (including possible subdirs)
    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}")
        return

    # Prefer CIFAR100 if present
    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    if df_all.empty:
        print(f"[WARN] No rows left after CIFAR100 filtering in {figset_dir}")
        return

    # Apply standard presentation policy: MAPL -> server only, others -> client
    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {figset_dir}")
        return

    # Keep only MAPL rows (just in case there is other stuff here)
    is_mapl = df_all["algorithm_display"].astype(str).str.lower().str.contains("mapl")
    df_all = df_all[is_mapl]
    if df_all.empty:
        print(f"[WARN] No MAPL rows found in {figset_dir}")
        return

    # Need lambda_consistency metadata
    if "lambda_consistency" not in df_all.columns:
        print(f"[WARN] lambda_consistency missing in df under {figset_dir}")
        return

    df_all = df_all.copy()
    df_all["lambda_consistency"] = pd.to_numeric(df_all["lambda_consistency"], errors="coerce")
    df_all = df_all.dropna(subset=["lambda_consistency"])

    if df_all.empty:
        print(f"[WARN] No rows with valid lambda_consistency in {figset_dir}")
        return

    if "_path" not in df_all.columns:
        print(f"[WARN] _path column missing in df under {figset_dir}")
        return

    # -------- Step 1: per-run curves (avg over clients) --------
    run_rows = []
    for (lam, path, seed), d in df_all.groupby(
        ["lambda_consistency", "_path", "seed"],
        dropna=False,
    ):
        if d.empty:
            continue
        sub = d[["iteration", "client_id", "accuracy"]].copy()
        s = _aggregate_run_mean_over_clients(sub)  # Series: index=iteration, values=accuracy
        if s.empty:
            continue
        run_rows.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "value": s.values.astype(float),
            "lambda_consistency": float(lam),
            "seed": seed,
            "_path": path,
        }))

    if not run_rows:
        print(f"[WARN] No per-run curves produced for {figset_dir}")
        return

    df_runs = pd.concat(run_rows, ignore_index=True)

    # If values look like probabilities (<=1), scale to %, per lambda group
    def _maybe_scale_group_lambda(g: pd.DataFrame) -> pd.DataFrame:
        vals = g["value"].to_numpy(dtype=float)
        g = g.copy()
        g["value"] = _maybe_scale_to_percent(vals)
        return g

    df_runs = df_runs.groupby(
        ["lambda_consistency"],
        group_keys=False,
    ).apply(_maybe_scale_group_lambda)

    # -------- Step 2: aggregate across seeds per lambda --------
    iter_stats = []
    for lam, g in df_runs.groupby("lambda_consistency"):
        mats = []
        for seed, d in g.groupby("seed"):
            d = d.sort_values("iter")
            vals = d["value"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            mats.append(vals)

        if not mats:
            continue

        T = min(len(a) for a in mats)
        if T == 0:
            continue
        arr = np.vstack([a[:T] for a in mats])  # [S, T]

        iters = np.sort(g["iter"].unique())[:T]
        means = np.nanmean(arr, axis=0)
        sems = (np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])) if arr.shape[0] > 1 else np.zeros(T)

        iter_stats.append(pd.DataFrame({
            "iter": iters.astype(int),
            "lambda_consistency": float(lam),
            "mean": means.astype(float),
            "sem": sems.astype(float),
            "n_seeds": int(arr.shape[0]),
        }))

    if not iter_stats:
        print(f"[WARN] No per-iteration stats available for {figset_dir}")
        return

    df_iter = pd.concat(iter_stats, ignore_index=True).sort_values(
        ["lambda_consistency", "iter"]
    )

    # -------- Step 3: save CSVs --------
    out_dir = out_root / "mapl_lambda"
    out_dir.mkdir(parents=True, exist_ok=True)

    iter_csv = out_dir / "mapl_lambda_iter.csv"
    df_iter.to_csv(_win_long_abs(iter_csv), index=False)
    print(f"[OK] Saved: {iter_csv}")

    df_final = _final_stats_for_mapl_lambda(df_iter)
    final_csv = out_dir / "mapl_lambda_final.csv"
    df_final.to_csv(_win_long_abs(final_csv), index=False)
    print(f"[OK] Saved: {final_csv}")

    # -------- Step 4: plot accuracy vs iteration (one curve per lambda) --------
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))

    # Nicely ordered λ values
    lambdas = sorted(df_iter["lambda_consistency"].unique().tolist())
    cmap = plt.get_cmap("tab10")

    for i, lam in enumerate(lambdas):
        d = df_iter[df_iter["lambda_consistency"] == lam].sort_values("iter")
        x = d["iter"].to_numpy()
        y = d["mean"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            linewidth=3.0,
            marker=None,
            label=f"λ={lam:g}",
            color=cmap(i % 10),
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)

    # Legend outside, one row, like client_scale
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if h:
        fig.legend(
            h, l,
            loc="upper center",
            ncol=min(6, len(lambdas)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
            fontsize=LEGEND_FONT_SIZE,
        )

    out_pdf = out_dir / "mapl_lambda.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"), help="Root results folder (contains figure-set folders)")
    ap.add_argument("--figdir", type=Path, default=Path("figures"), help="Where to save PDFs/CSVs")
    ap.add_argument("--inspect", action="store_true", help="Print detected paths/keys while loading")
    ap.add_argument("--lambda_folder", type=str, default="examine_lambda_for_ditto")
    #ap.add_argument("--alpha", type=int, default=5)
    ap.add_argument("--facet_client_net_dir", type=Path, default=Path("results/diff_clients_nets"),
                    help="Folder whose immediate subfolders are algorithm folders; used to facet by client_net_type._value_")
    ap.add_argument("--global_data_size_folder", type=str, default="global_data_size",
                    help="Folder under --root where each immediate subfolder represents one curve/ratio.")
    ap.add_argument("--client_scale_folder", type=str, default="client_scale",
                    help="Folder under --root containing runs with different summary[num_clients].")

    args = ap.parse_args()

    # Global-data-size: SERVER-only, labels are numeric ratios (+CSV)
    figure_global_data_size_oneplot(args.root / args.global_data_size_folder,args.figdir,inspect=args.inspect,use_server_measure=True
    )

    # NEW: Client-scale oneplot (+CSVs with per-iter and final stats)
    figure_client_scale_oneplot(
        args.root / args.client_scale_folder,
        args.figdir,
        inspect=args.inspect
    )

    # Comparison figures (+CSVs): client for non-MAPL + MAPL server-only (as "MAPL")
    figure_diff_clients_nets(args.root / "diff_clients_nets", args.figdir, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "diff_benchmarks_05", args.figdir, alpha_value=5, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "diff_benchmarks_100", args.figdir, alpha_value=100, inspect=args.inspect)

    figure_diff_benchmarks(args.root / "diff_benchmarks_1", args.figdir, alpha_value=1, inspect=args.inspect)
    figure_by_client_net_type_value(args.root / "same_client_nets", args.figdir, inspect=args.inspect)


    figure_temp_serverinput_oneplot(
    args.root / "temp_serverinput",
    args.figdir,
    inspect=args.inspect,
    )
    figure_mapl_lambda_oneplot(
        args.root / "mapl_lambda",
        args.figdir,
        inspect=args.inspect,
    )
if __name__ == "__main__":
    main()
