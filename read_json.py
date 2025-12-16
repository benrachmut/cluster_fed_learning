#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig_algorithms_avg_accuracy.py

This version:
  - DOES NOT create any *_iter.csv files (only final CSVs).
  - DOES NOT create any output subfolders under --figdir.
    All PDFs/CSVs are written directly into --figdir.

Figures / outputs (PDF + CSV where applicable):
  - figures/client_net_type_value.pdf + figures/client_net_type_value.csv
  - figures/diff_clients_nets.pdf + figures/diff_clients_nets.csv
  - figures/diff_benchmarks_05.pdf + figures/diff_benchmarks_05.csv
  - figures/diff_benchmarks_100.pdf + figures/diff_benchmarks_100.csv
  - figures/diff_benchmarks_1.pdf + figures/diff_benchmarks_1.csv
  - figures/global_data_size_by_server_data_ratio_{client|server}.pdf + .csv
  - figures/client_scale.pdf + figures/client_scale_final.csv
  - figures/temp_serverinput.pdf + figures/temp_serverinput_final.csv
  - figures/mapl_lambda.pdf + figures/mapl_lambda_final.csv
  - figures/mapl_lambda_and_temp.pdf
      + figures/mapl_lambda_and_temp_lambda_final.csv
      + figures/mapl_lambda_and_temp_temp_final.csv
  - figures/client_scale_and_global_data_size_{suffix}.pdf
      + figures/client_scale_and_global_data_size_client_scale_final.csv
      + figures/client_scale_and_global_data_size_global_data_final.csv

NEW:
  - figures/diff_server_nets.pdf + figures/diff_server_nets_final.csv
    (curves keyed by summary["server_net_type"]["_value_"], MAPL server-only)

Also:
  - Forces all x-axes to [0, 9].
  - MAPL plotting policy: use MAPL server curve only, labeled "MAPL".
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


# --------------------- paired t-test helpers (run-level) ---------------------

def _ttest_rel_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired t-test p-value. Returns NaN if insufficient samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if a.size < 2:
        return float("nan")
    try:
        import scipy.stats as _st  # type: ignore
        return float(_st.ttest_rel(a, b, nan_policy="omit").pvalue)
    except Exception:
        # Fallback: normal approximation on paired differences (OK for moderate n)
        d = a - b
        n = d.size
        if n < 2:
            return float("nan")
        sd = float(np.std(d, ddof=1))
        if sd == 0.0:
            return 0.0
        t = float(np.mean(d) / (sd / np.sqrt(n)))
        # two-sided normal approx
        z = abs(t)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        return float(p)

def _add_pvalues_vs_best(df_stats: pd.DataFrame, *, group_cols: List[str], id_col: str, run_key_cols: List[str], value_col: str) -> pd.DataFrame:
    """
    For each group (defined by group_cols), compute paired t-test p-values comparing each id_col level
    to the best (highest-mean) id_col level, using run-level values aligned by run_key_cols.
    Expects df_stats to already contain summary rows for each id within each group.
    """
    if df_stats is None or df_stats.empty:
        return df_stats
    df_stats = df_stats.copy()
    df_stats["compared_to"] = ""
    df_stats["p_value"] = np.nan

    # We'll compute p-values from a raw run-level table stored on df_stats via attrs (if present)
    run_table = df_stats.attrs.get("_run_table", None)
    if run_table is None or not isinstance(run_table, pd.DataFrame) or run_table.empty:
        return df_stats

    for _, grp_idx in df_stats.groupby(group_cols, dropna=False).groups.items():
        sub = df_stats.loc[grp_idx].copy()
        # pick best by mean
        if sub.empty:
            continue
        best_row = sub.sort_values("mean", ascending=False).iloc[0]
        best_id = best_row[id_col]

        # build best series keyed by run
        rt = run_table.copy()
        # restrict to same group
        for c in group_cols:
            rt = rt[rt[c] == best_row[c]]
        if rt.empty:
            continue

        best_rt = rt[rt[id_col] == best_id]
        if best_rt.empty:
            continue
        best_rt = best_rt[run_key_cols + [value_col]].dropna()
        best_rt = best_rt.rename(columns={value_col: "best_val"})

        for i in grp_idx:
            cur_id = df_stats.at[i, id_col]
            if cur_id == best_id:
                df_stats.at[i, "compared_to"] = ""
                df_stats.at[i, "p_value"] = np.nan
                continue
            cur_rt = rt[rt[id_col] == cur_id]
            if cur_rt.empty:
                df_stats.at[i, "compared_to"] = str(best_id)
                df_stats.at[i, "p_value"] = np.nan
                continue
            cur_rt = cur_rt[run_key_cols + [value_col]].dropna()
            cur_rt = cur_rt.rename(columns={value_col: "cur_val"})

            merged = pd.merge(best_rt, cur_rt, on=run_key_cols, how="inner")
            if merged.empty:
                p = float("nan")
            else:
                p = _ttest_rel_pvalue(merged["best_val"].to_numpy(), merged["cur_val"].to_numpy())
            df_stats.at[i, "compared_to"] = str(best_id)
            df_stats.at[i, "p_value"] = float(p)

    return df_stats
FINAL_ITER = 9  # used only for "final" stats

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

TITLE_MAP = {
    "rndNet":    "{AlexNet, ResNet, MobileNet, SqueezeNet}",
    "rndStrong": "{AlexNet, ResNet}",
    "rndWeak":   "{MobileNet, SqueezeNet}",
    "AlexMobile":"{AlexNet, MobileNet}",
    "AlexSqueeze":"{AlexNet, SqueezeNet}",
    "ResMobile":"{ResNet, MobileNet}",
    "ResNetSqueeze":"{ResNet, SqueezeNet}",
    "AlexMobileResnet": "{MobileNet,ResNet, AlexNet}",
}

CLIENT_NET_SUBFIG_ORDER = [
    "rndNet",
    "AlexMobileResnet",
    "rndStrong",
    "AlexMobile",
    "AlexSqueeze",
    "ResMobile",
    "ResNetSqueeze",
    "rndWeak",
]

TITLE_NET_MAP = {
    "ALEXNET": "AlexNet",
    "MobileNet": "MobileNet",
    "RESNET": "ResNet",
    "SqueezeNet": "SqueezeNet",
}

def _tab10_colors_excluding_blue():
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(10)]
    # tab10[0] is the default blue — exclude it
    return colors[1:]


def _make_color_map(keys, *, special_key=None, special_color="tab:blue"):
    """
    Build a mapping key -> color such that:
      - special_key (if provided and present) gets special_color (blue)
      - all other keys get distinct colors, none of which is the tab10 blue
    """
    keys = list(keys)
    palette = _tab10_colors_excluding_blue()

    cmap = {}
    if special_key is not None and special_key in keys:
        cmap[special_key] = special_color

    j = 0
    for k in keys:
        if k == special_key:
            continue
        cmap[k] = palette[j % len(palette)]
        j += 1
    return cmap

def _color_override_global_data_ratio(label: str):
    """
    global_data_size color override:
      - ratio == 1.0 must be tab:blue
    """
    try:
        if float(label) == 1.0:
            return "tab:blue"
    except Exception:
        pass
    return None


def _color_override_server_net(sn: str):
    """
    diff_server_nets color override:
      - any server net containing 'vgg' -> tab:blue
    """
    if sn is None:
        return None
    if "vgg" in str(sn).strip().lower():
        return "tab:blue"
    return None

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
            "server_net_type_value",
            "alpha_dich",
            "lambda_ditto",
            "lambda_consistency",
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

# ---- dataset-aware accuracy map selectors (Top-1 vs Top-5) ----

def _is_top5_dataset(ds: Optional[str]) -> bool:
    if not ds: return False
    dsl = str(ds).lower()
    return ("imagenetr" in dsl) or ("tinyimagenet" in dsl)

def find_client_accuracy_map_topk(root: Json, k: int):
    v, p = find_exact_key_map(root, f"client_accuracy_per_client_{k}")
    if v is not None: return v, p
    c: List[Tuple[Tuple[str, ...], Any]] = []
    for p, v in walk(root):
        if looks_like_client_accuracy_map(v) and any("client" in seg.lower() for seg in p):
            c.append((p, v))
    if c:
        def score(pv):
            p,_ = pv; s=0
            if any("summary" in seg.lower() for seg in p): s+=2
            if any("client" in seg.lower() for seg in p): s+=1
            if any("5" in seg for seg in p): s+=1
            return s
        c.sort(key=score, reverse=True)
        return c[0][1], c[0][0]
    return None, None

def find_server_accuracy_map_topk(root: Json, k: int):
    for key in (f"server_accuracy_per_client_{k}_max", f"server_accuracy_per_client_{k}"):
        v, p = find_exact_key_map(root, key)
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
            if any(str(k) in seg for seg in p): s+=1
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

def find_num_clients(root: Json):
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

def find_lambda_consistency(root: Json):
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

        ds_name, _            = find_dataset_selected_name(raw)
        alg_name, _           = find_algorithm_name(raw)
        if not alg_name:
            alg_name = alg_hint or dir_path.name
        seed_num, _           = find_seed_num(raw)

        use_top5 = _is_top5_dataset(ds_name)
        topk = 5 if use_top5 else 1

        client_map, _         = find_client_accuracy_map_topk(raw, topk)
        server_map, _         = find_server_accuracy_map_topk(raw, topk)

        server_net_val, _     = find_server_net_type_value(raw)
        client_net_name, _    = find_client_net_type_name(raw)
        client_net_value, _   = find_client_net_type_value(raw)
        alpha_dich, _         = find_alpha_dich(raw)
        lambda_ditto, _       = find_lambda_ditto(raw)
        server_data_ratio, _  = find_server_data_ratio(raw, fallback_from=p)
        num_clients, _        = find_num_clients(raw)
        lambda_consistency, _ = find_lambda_consistency(raw)

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
                        "server_net_type_value": server_net_val,
                        "alpha_dich": alpha_dich,
                        "lambda_ditto": lambda_ditto,
                        "server_data_ratio": server_data_ratio,
                        "num_clients": num_clients,
                        "server_input_tech_name": server_input_tech_name,
                        "distill_temperature": distill_temperature,
                        "lambda_consistency": lambda_consistency,
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
                        "server_net_type_value": server_net_val,
                        "alpha_dich": alpha_dich,
                        "lambda_ditto": lambda_ditto,
                        "server_data_ratio": server_data_ratio,
                        "num_clients": num_clients,
                        "server_input_tech_name": server_input_tech_name,
                        "distill_temperature": distill_temperature,
                        "lambda_consistency": lambda_consistency,
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
        "server_net_type_value",
        "alpha_dich",
        "lambda_ditto",
        "lambda_consistency",
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
    return "-"

def _normalize_mapl_presentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop MAPL client rows (MAPL_C-*) and rename MAPL_S-* -> MAPL.
    """
    if df is None or df.empty or "algorithm_display" not in df.columns:
        return df
    df = df.copy()
    algdisp = df["algorithm_display"].astype(str)

    mask_c = algdisp.str.startswith("MAPL_C-")
    if mask_c.any():
        df = df[~mask_c]

    mask_s = df["algorithm_display"].astype(str).str.startswith("MAPL_S-")
    if mask_s.any():
        df.loc[mask_s, "algorithm_display"] = "MAPL"

    return df

# --------------------- CSV stats helpers ---------------------


def _final_stats_for_panel(df_panel: pd.DataFrame, *, subfigure: str, final_iter: int = FINAL_ITER) -> pd.DataFrame:
    """
    Compute final stats at a fixed iteration (default: 9).
    Stats are computed across RUNS (unique (_path, seed)), where each run's value
    is the mean accuracy across clients at that iteration.

    Also adds:
      - compared_to (best method in this panel)
      - p_value (paired t-test vs best, across shared runs)
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_rel

    if df_panel is None or df_panel.empty:
        return pd.DataFrame(columns=["subfigure", "algorithm", "iteration", "mean", "std", "sem", "n", "compared_to", "p_value"])

    d = df_panel.copy()
    d["iteration"] = pd.to_numeric(d["iteration"], errors="coerce")
    d = d[d["iteration"] == final_iter]
    if d.empty:
        return pd.DataFrame(columns=["subfigure", "algorithm", "iteration", "mean", "std", "sem", "n", "compared_to", "p_value"])

    # Per-run value: average across clients within each independent run
    run_vals = (
        d.groupby(["algorithm_display", "_path", "seed"], dropna=False)["accuracy"]
         .mean()
         .reset_index()
         .rename(columns={"accuracy": "run_acc"})
    )

    # Build per-algorithm mapping from run key -> value
    run_key = ["_path", "seed"]
    alg_to_series = {}
    for alg, g in run_vals.groupby("algorithm_display", dropna=False):
        gg = g.copy()
        gg["_path"] = gg["_path"].astype(str)
        gg["seed"] = gg["seed"]
        alg_to_series[str(alg)] = gg.set_index(run_key)["run_acc"]

    # Determine best algorithm by mean over its runs
    alg_means = {alg: float(s.mean()) for alg, s in alg_to_series.items() if len(s) > 0}
    if not alg_means:
        return pd.DataFrame(columns=["subfigure", "algorithm", "iteration", "mean", "std", "sem", "n", "compared_to", "p_value"])

    best_alg = max(alg_means.items(), key=lambda kv: kv[1])[0]
    best_series = alg_to_series[best_alg]

    out_rows = []
    for alg, s in alg_to_series.items():
        vals = pd.to_numeric(s.values, errors="coerce")
        vals = vals[~np.isnan(vals)]
        n = int(vals.size)
        mean = float(np.mean(vals)) if n > 0 else np.nan
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0

        # paired t-test vs best on intersection of runs
        compared_to = best_alg
        p_value = np.nan
        if alg != best_alg:
            common_idx = s.index.intersection(best_series.index)
            if len(common_idx) >= 2:
                a = s.loc[common_idx].astype(float).values
                b = best_series.loc[common_idx].astype(float).values
                # compare (best - alg) or (alg - best) doesn't matter for p-value
                _, p = ttest_rel(b, a, nan_policy="omit")
                p_value = float(p)

        out_rows.append({
            "subfigure": subfigure,
            "algorithm": alg,
            "iteration": int(final_iter),
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
            "compared_to": compared_to,
            "p_value": p_value if alg != best_alg else np.nan,
        })

    return pd.DataFrame(out_rows, columns=["subfigure", "algorithm", "iteration", "mean", "std", "sem", "n", "compared_to", "p_value"])


def _save_stats_csv(rows: List[pd.DataFrame], outfile_pdf: Path):
    df = _concat_or_empty(rows)
    csv_path = outfile_pdf.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df = pd.DataFrame(columns=["subfigure", "algorithm", "iteration", "mean", "std", "sem", "n"])
    df.to_csv(_win_long_abs(csv_path), index=False)
    print(f"[OK] Saved CSV: {csv_path}")

def _save_df_csv(df: pd.DataFrame, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_win_long_abs(out_csv), index=False)
    print(f"[OK] Saved CSV: {out_csv}")

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

def _force_x_axes_0_to_9(fig):
    for ax in fig.get_axes():
        try:
            ax.set_xlim(0, 9)
        except Exception:
            pass

def _apply_measure_filter_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "measure" not in df.columns or "algorithm_display" not in df.columns:
        return df
    df = df.copy()
    algdisp = df["algorithm_display"].astype(str)
    is_mapl = algdisp.str.contains(r"^MAPL", regex=True)
    keep_non_mapl = (~is_mapl) & (df["measure"].astype(str) == PLOT_MEASURE_BASE)
    keep_mapl     = (is_mapl) & (df["measure"].astype(str) == "server")
    out = df[keep_non_mapl | keep_mapl]
    return _normalize_mapl_presentation(out)

# --------------------------------------------------------------------
# diff_benchmarks
# --------------------------------------------------------------------
def figure_diff_benchmarks(figset_dir: Path, out_root: Path, *, alpha_value: int, inspect: bool):
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
        ax.set_ylabel("Top-5 Accuracy" if _is_top5_dataset(choice) else "Top-1 Accuracy")
        ax.grid(False)

    _force_x_axes_0_to_9(fig)

    fig.supxlabel("Iteration", fontsize=BASE_FONT_SIZE)
    h, l = _legend_from_axes_row(axes)
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

# --------------------------------------------------------------------
# diff_clients_nets
# --------------------------------------------------------------------
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

    def _mode_or_none(series):
        if series is None:
            return None
        s = series.dropna().astype(str)
        if s.empty:
            return None
        return s.value_counts().index[0]

    def _subfig_sort_key(item):
        sf, df = item
        key_name = None
        if "client_net_type_name" in df.columns:
            key_name = _mode_or_none(df["client_net_type_name"])
        if (not key_name) and ("client_net_type_value" in df.columns):
            key_name = _mode_or_none(df["client_net_type_value"])
        if key_name in CLIENT_NET_SUBFIG_ORDER:
            return (CLIENT_NET_SUBFIG_ORDER.index(key_name), key_name or sf.name)
        return (len(CLIENT_NET_SUBFIG_ORDER), sf.name)

    loaded.sort(key=_subfig_sort_key)

    n = len(loaded)
    ncols = max(1, min(max_cols, n))
    nrows = math.ceil(n / ncols)

    fig_w = max(5.0 * ncols, 5.0)
    fig_h = max(4.5 * nrows, 4.5)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
        squeeze=False
    )
    axes = axes.flatten()

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
        ax.tick_params(labelleft=True)

    for ax in axes[n:]:
        ax.set_visible(False)

    _force_x_axes_0_to_9(fig)

    fig.supxlabel("Iteration", fontsize=BASE_FONT_SIZE)
    fig.supylabel("Top-1 Accuracy", fontsize=BASE_FONT_SIZE)

    first_ax = next((a for a in axes[:n] if a.get_visible()), None)
    h, l = (first_ax.get_legend_handles_labels() if first_ax else ([], []))

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

# --------------------------------------------------------------------
# by client_net_type_value
# --------------------------------------------------------------------
def figure_by_client_net_type_value(figset_dir: Path, out_root: Path, *, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}"); return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

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
        ax.grid(False)
        ax.set_ylim(ymin, ymax)

    _force_x_axes_0_to_9(fig)

    fig.supxlabel("Iteration", fontsize=BASE_FONT_SIZE)
    fig.supylabel("Top-1 Accuracy", fontsize=BASE_FONT_SIZE)

    h, l = _legend_from_axes_row(axes)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if h:
        fig.legend(
            h, l,
            loc="upper center",
            ncol=len(h),
            frameon=False,
            bbox_to_anchor=(0.5, 0.96),
            fontsize=LEGEND_FONT_SIZE,
        )

    outfile = out_root / "client_net_type_value.pdf"
    _savefig_longpath(fig, outfile); plt.close(fig)
    print(f"[OK] Saved: {outfile}")
    _save_stats_csv(stats_rows, outfile)

# --------------------- Global-data-size (ONE plot) ---------------------

def figure_global_data_size_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool, use_server_measure: bool = False):
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
        override = _color_override_global_data_ratio(lab)
        ax.plot(
            g["iteration"], g["avg_accuracy"],
            linewidth=3.0, marker=None,
            label=lab,
            color=(override if override is not None else cmap(i % 10)),
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)

    _force_x_axes_0_to_9(fig)

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

# --------------------- client_scale (ONE plot; one curve per num_clients) ---------------------

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
    g = df_run.groupby("iteration", dropna=False)["accuracy"].mean().sort_index()
    return g


def _final_stats_for_client_scale_from_iter(df_iter: pd.DataFrame, *, final_iter: int = FINAL_ITER) -> pd.DataFrame:
    """
    Final stats for client_scale.

    Independence policy:
      - Each run is identified by (_path, seed) within each num_clients.
      - We compute stats over run-level final values, not over client-level rows.

    Adds p-values comparing each num_clients setting to the best-performing num_clients
    (paired by common run keys when possible).
    """
    cols_out = ["num_clients","iteration","mean","std","sem","n","compared_to","p_value"]
    d = df_iter.copy()
    d["iter"] = pd.to_numeric(d["iter"], errors="coerce")
    d = d[d["iter"] == final_iter].copy()
    if d.empty:
        return pd.DataFrame(columns=cols_out)

    # Run-level final value is already stored in df_iter as run_value per run per iter,
    # but df_iter may contain multiple rows per run (if upstream duplicated). Deduplicate by run keys.
    run_table = (
        d.groupby(["num_clients","_path","seed"], dropna=False)["run_value"]
         .mean()
         .reset_index()
         .rename(columns={"run_value":"run_value"})
    )

    out = []
    for nc, g in run_table.groupby("num_clients", dropna=False):
        vals = pd.to_numeric(g["run_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        n = int(vals.size)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        out.append({
            "num_clients": int(nc) if pd.notna(nc) else None,
            "iteration": int(final_iter),
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
        })

    df_stats = pd.DataFrame(out, columns=["num_clients","iteration","mean","std","sem","n"])
    if df_stats.empty:
        return pd.DataFrame(columns=cols_out)

    # p-values vs best num_clients
    df_stats.attrs["_run_table"] = run_table.assign(subfigure="client_scale").rename(columns={"num_clients":"id"})
    df_stats = df_stats.rename(columns={"num_clients":"id"}).assign(subfigure="client_scale")

    df_stats = _add_pvalues_vs_best(
        df_stats,
        group_cols=["subfigure"],
        id_col="id",
        run_key_cols=["_path","seed"],
        value_col="run_value",
    )

    # restore names
    df_stats = df_stats.rename(columns={"id":"num_clients"})
    return df_stats[cols_out]


def figure_client_scale_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)"); return

    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}"); return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {figset_dir}"); return

    if "num_clients" not in df_all.columns or df_all["num_clients"].isna().all():
        print(f"[WARN] No summary[num_clients] detected in {figset_dir}"); return

    df_all = df_all.copy()
    df_all["num_clients"] = pd.to_numeric(df_all["num_clients"], errors="coerce")
    df_all = df_all.dropna(subset=["num_clients"])

    # Build per-run curve (mean over clients), then compute final stats (iter=9) over runs
    run_rows = []
    for (path, nc, seed), d in df_all.groupby(["_path","num_clients","seed"], dropna=False):
        if d.empty:
            continue
        sub = d[["iteration","client_id","accuracy"]].copy()
        s = _aggregate_run_mean_over_clients(sub)
        if s.empty:
            continue
        svals = _maybe_scale_to_percent(s.values.astype(float))
        run_rows.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "run_value": svals,
            "num_clients": int(nc),
            "seed": seed,
            "_path": path
        }))

    if not run_rows:
        print(f"[WARN] No per-run curves produced for {figset_dir}"); return

    df_runs = pd.concat(run_rows, ignore_index=True)

    # Per-iteration mean curve (for plotting)
    df_iter = (
        df_runs.groupby(["num_clients","iter"], dropna=False)["run_value"]
              .mean()
              .reset_index()
              .rename(columns={"run_value":"mean"})
              .sort_values(["num_clients","iter"])
    )

    # Final CSV only
    df_final = _final_stats_for_client_scale_from_iter(df_runs)
    final_csv = out_root / "client_scale_final.csv"
    _save_df_csv(df_final, final_csv)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    for nc in sorted(df_iter["num_clients"].unique().tolist()):
        d = df_iter[df_iter["num_clients"] == nc].sort_values("iter")
        ax.plot(d["iter"], d["mean"], linewidth=3.0, label=f"{int(nc)} clients")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)
    _force_x_axes_0_to_9(fig)

    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)),
                   frameon=False, bbox_to_anchor=(0.5, 0.96),
                   fontsize=LEGEND_FONT_SIZE)

    out_pdf = out_root / "client_scale.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

# --------------------- temp_serverinput (ONE plot) + FINAL CSV only ---------------------


def _final_stats_for_temp_serverinput_from_runs(df_runs: pd.DataFrame, *, final_iter: int = FINAL_ITER) -> pd.DataFrame:
    """
    Final stats for (server_input_tech_name, distill_temperature).

    Independence policy:
      - Each run is identified by (_path, seed) within each (tech, temp).
      - Stats are computed across runs (seeds), not across clients.

    Adds p-values comparing each (tech,temp) combo to the best combo (paired by common run keys).
    """
    cols_out = [
        "server_input_tech_name","distill_temperature","iteration","mean","std","sem","n","compared_to","p_value"
    ]

    d = df_runs.copy()
    d["iter"] = pd.to_numeric(d["iter"], errors="coerce")
    d = d[d["iter"] == final_iter].copy()
    if d.empty:
        return pd.DataFrame(columns=cols_out)

    # Deduplicate to one run-level value per (_path, seed, tech, temp)
    run_table = (
        d.groupby(["server_input_tech_name","distill_temperature","_path","seed"], dropna=False)["run_value"]
         .mean()
         .reset_index()
    )

    out = []
    keys = ["server_input_tech_name","distill_temperature"]
    for (tech, temp), g in run_table.groupby(keys):
        vals = pd.to_numeric(g["run_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        n = int(vals.size)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        out.append({
            "server_input_tech_name": str(tech),
            "distill_temperature": float(temp),
            "iteration": int(final_iter),
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
        })

    df_stats = pd.DataFrame(out, columns=["server_input_tech_name","distill_temperature","iteration","mean","std","sem","n"])
    if df_stats.empty:
        return pd.DataFrame(columns=cols_out)

    # Create an id label for comparison
    df_stats = df_stats.assign(subfigure="temp_serverinput")
    run_table = run_table.assign(subfigure="temp_serverinput")
    run_table["id"] = run_table.apply(lambda r: f"T={r['distill_temperature']:g}, {r['server_input_tech_name']}", axis=1)
    df_stats["id"] = df_stats.apply(lambda r: f"T={r['distill_temperature']:g}, {r['server_input_tech_name']}", axis=1)

    df_stats.attrs["_run_table"] = run_table.rename(columns={"run_value":"run_value"})
    df_stats = _add_pvalues_vs_best(
        df_stats,
        group_cols=["subfigure"],
        id_col="id",
        run_key_cols=["_path","seed"],
        value_col="run_value",
    )

    # Map compared_to back to readable id already
    df_stats["compared_to"] = df_stats["compared_to"].astype(str)
    return df_stats[cols_out]


def figure_temp_serverinput_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)")
        return

    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}")
        return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    if df_all.empty:
        print(f"[WARN] No rows left after CIFAR100 filtering in {figset_dir}")
        return

    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {figset_dir}")
        return

    if "server_input_tech_name" not in df_all.columns or "distill_temperature" not in df_all.columns:
        print(f"[WARN] server_input_tech_name/distill_temperature missing in df under {figset_dir}")
        return

    df_all = df_all.copy()
    df_all["server_input_tech_name"] = df_all["server_input_tech_name"].astype(str)
    df_all["distill_temperature"] = pd.to_numeric(df_all["distill_temperature"], errors="coerce")
    df_all = df_all.dropna(subset=["distill_temperature"])
    df_all = df_all[df_all["server_input_tech_name"].str.lower().ne("none")]
    if df_all.empty:
        print(f"[WARN] No rows with valid server_input_tech_name and distill_temperature in {figset_dir}")
        return

    run_rows = []
    for (tech, temp, path, seed), d in df_all.groupby(
        ["server_input_tech_name", "distill_temperature", "_path", "seed"],
        dropna=False,
    ):
        if d.empty:
            continue
        sub = d[["iteration", "client_id", "accuracy"]].copy()
        s = _aggregate_run_mean_over_clients(sub)
        if s.empty:
            continue
        run_rows.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "run_value": _maybe_scale_to_percent(s.values.astype(float)),
            "server_input_tech_name": tech,
            "distill_temperature": float(temp),
            "seed": seed,
            "_path": path,
        }))

    if not run_rows:
        print(f"[WARN] No per-run curves produced for {figset_dir}")
        return

    df_runs = pd.concat(run_rows, ignore_index=True)

    # Final CSV only
    df_final = _final_stats_for_temp_serverinput_from_runs(df_runs)
    _save_df_csv(df_final, out_root / "temp_serverinput_final.csv")

    # Per-iteration mean curve (for plotting)
    df_iter = (
        df_runs.groupby(["server_input_tech_name","distill_temperature","iter"], dropna=False)["run_value"]
              .mean()
              .reset_index()
              .rename(columns={"run_value":"mean"})
              .sort_values(["distill_temperature","server_input_tech_name","iter"])
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))

    ymin = 15
    ymax = 50
    ax.set_ylim(ymin, ymax)

    cmap = plt.get_cmap("tab10")

    temps = sorted(df_iter["distill_temperature"].dropna().unique().tolist())
    temp_color: Dict[float, Any] = {t: cmap(i % 10) for i, t in enumerate(temps)}

    def _ls_for_tech(tech: str) -> str:
        tl = (tech or "").strip().lower()
        if "mean" in tl:
            return ":"
        if "max" in tl:
            return "-"
        if "median" in tl or "med" in tl:
            return "--"
        return "-"

    def _tech_order(tech: str) -> int:
        tl = (tech or "").strip().lower()
        if "max" in tl:
            return 0
        if "mean" in tl:
            return 1
        if "median" in tl or "med" in tl:
            return 2
        return 3

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
        label = f"T={temp:g}, {tech}"
        ax.plot(x, y, linewidth=3.0, marker=None, label=label, color=color, linestyle=ls)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)

    _force_x_axes_0_to_9(fig)

    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    if h:
        fig.legend(
            h, l,
            loc="center left",
            bbox_to_anchor=(0.96, 0.5),
            ncol=1,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            borderaxespad=0.3,
        )

    out_pdf = out_root / "temp_serverinput.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

# --------------------- MAPL lambda (ONE plot) + FINAL CSV only ---------------------


def _final_stats_for_mapl_lambda_from_runs(df_runs: pd.DataFrame, *, final_iter: int = FINAL_ITER) -> pd.DataFrame:
    """
    Final stats for lambda_consistency sweep (MAPL/COSMOS lambda).

    Independence policy:
      - Each run is identified by (_path, seed) within each lambda_consistency.
      - Stats are computed across runs (seeds), not across clients.

    Adds p-values comparing each lambda to the best lambda (paired by common run keys).
    """
    cols_out = ["lambda_consistency","iteration","mean","std","sem","n","compared_to","p_value"]

    d = df_runs.copy()
    d["iter"] = pd.to_numeric(d["iter"], errors="coerce")
    d = d[d["iter"] == final_iter].copy()
    if d.empty:
        return pd.DataFrame(columns=cols_out)

    # One run-level final value per (_path, seed, lambda)
    run_table = (
        d.groupby(["lambda_consistency","_path","seed"], dropna=False)["run_value"]
         .mean()
         .reset_index()
    )

    out = []
    for lam, g in run_table.groupby("lambda_consistency"):
        vals = pd.to_numeric(g["run_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        n = int(vals.size)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        out.append({
            "lambda_consistency": float(lam) if pd.notna(lam) else None,
            "iteration": int(final_iter),
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
        })

    df_stats = pd.DataFrame(out, columns=["lambda_consistency","iteration","mean","std","sem","n"])
    if df_stats.empty:
        return pd.DataFrame(columns=cols_out)

    df_stats = df_stats.assign(subfigure="mapl_lambda", id=df_stats["lambda_consistency"].apply(lambda x: f"λ={x:g}" if pd.notna(x) else "λ=NA"))
    run_table = run_table.assign(subfigure="mapl_lambda")
    run_table["id"] = run_table["lambda_consistency"].apply(lambda x: f"λ={x:g}" if pd.notna(x) else "λ=NA")

    df_stats.attrs["_run_table"] = run_table.rename(columns={"run_value":"run_value"})
    df_stats = _add_pvalues_vs_best(
        df_stats,
        group_cols=["subfigure"],
        id_col="id",
        run_key_cols=["_path","seed"],
        value_col="run_value",
    )

    # keep lambda_consistency numeric in output; compared_to stays as string label
    return df_stats[cols_out]


def figure_mapl_lambda_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool):
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)")
        return

    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}")
        return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]
    if df_all.empty:
        print(f"[WARN] No rows left after CIFAR100 filtering in {figset_dir}")
        return

    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {figset_dir}")
        return

    is_mapl = df_all["algorithm_display"].astype(str).str.lower().str.contains("mapl")
    df_all = df_all[is_mapl]
    if df_all.empty:
        print(f"[WARN] No MAPL rows found in {figset_dir}")
        return

    if "lambda_consistency" not in df_all.columns:
        print(f"[WARN] lambda_consistency missing in df under {figset_dir}")
        return

    df_all = df_all.copy()
    df_all["lambda_consistency"] = pd.to_numeric(df_all["lambda_consistency"], errors="coerce")
    df_all = df_all.dropna(subset=["lambda_consistency"])
    if df_all.empty:
        print(f"[WARN] No rows with valid lambda_consistency in {figset_dir}")
        return

    run_rows = []
    for (lam, path, seed), d in df_all.groupby(["lambda_consistency", "_path", "seed"], dropna=False):
        if d.empty:
            continue
        sub = d[["iteration", "client_id", "accuracy"]].copy()
        s = _aggregate_run_mean_over_clients(sub)
        if s.empty:
            continue
        run_rows.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "run_value": _maybe_scale_to_percent(s.values.astype(float)),
            "lambda_consistency": float(lam),
            "seed": seed,
            "_path": path,
        }))

    if not run_rows:
        print(f"[WARN] No per-run curves produced for {figset_dir}")
        return

    df_runs = pd.concat(run_rows, ignore_index=True)

    # Final CSV only
    df_final = _final_stats_for_mapl_lambda_from_runs(df_runs)
    _save_df_csv(df_final, out_root / "mapl_lambda_final.csv")

    # Per-iteration mean curve (for plotting)
    df_iter = (
        df_runs.groupby(["lambda_consistency","iter"], dropna=False)["run_value"]
              .mean()
              .reset_index()
              .rename(columns={"run_value":"mean"})
              .sort_values(["lambda_consistency","iter"])
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    lambdas = sorted(df_iter["lambda_consistency"].unique().tolist())
    cmap = plt.get_cmap("tab10")

    for i, lam in enumerate(lambdas):
        d = df_iter[df_iter["lambda_consistency"] == lam].sort_values("iter")
        ax.plot(d["iter"], d["mean"], linewidth=3.0, marker=None, label=f"λ={lam:g}", color=cmap(i % 10))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)

    _force_x_axes_0_to_9(fig)

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

    out_pdf = out_root / "mapl_lambda.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

# --------------------- NEW: diff_server_nets (ONE plot) + FINAL CSV only ---------------------

def _final_stats_for_diff_server_nets_from_runs(df_runs: pd.DataFrame, *, final_iter: int = FINAL_ITER) -> pd.DataFrame:
    d = df_runs.copy()
    d["iter"] = pd.to_numeric(d["iter"], errors="coerce")
    d = d[d["iter"] == final_iter].copy()
    if d.empty:
        return pd.DataFrame(columns=["server_net_type_value","iteration","mean","std","sem","n"])

    out = []
    for sn, g in d.groupby("server_net_type_value"):
        vals = pd.to_numeric(g["run_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        n = int(vals.size)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        out.append({
            "server_net_type_value": str(sn),
            "iteration": int(final_iter),
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
        })
    return pd.DataFrame(out, columns=["server_net_type_value","iteration","mean","std","sem","n"])

def figure_diff_server_nets_oneplot(figset_dir: Path, out_root: Path, *, inspect: bool):
    """
    results/diff_server_nets/
      <any subfolders or jsons>

    Curves are keyed by summary["server_net_type"]["_value_"] (stored as df["server_net_type_value"]).
    Uses MAPL policy: server-measure only, labeled MAPL.
    """
    if not figset_dir.exists():
        print(f"[SKIP] {figset_dir} (missing)")
        return

    df_all = _load_any_jsons_under(figset_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {figset_dir}")
        return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {figset_dir}")
        return

    # Ensure MAPL server-only (extra guard)
    df_all = df_all[df_all["measure"].astype(str) == "server"].copy()
    df_all = df_all[df_all["algorithm_display"].astype(str) == "MAPL"].copy()
    if df_all.empty:
        print(f"[WARN] No MAPL server rows under {figset_dir}")
        return

    if "server_net_type_value" not in df_all.columns:
        print(f"[WARN] server_net_type_value missing (check load_rows_from_dir additions).")
        return

    df_all["server_net_type_value"] = df_all["server_net_type_value"].astype(str)
    df_all = df_all[~df_all["server_net_type_value"].str.lower().isin(["none", "nan", ""])].copy()
    if df_all.empty:
        print(f"[WARN] No valid server_net_type_value rows under {figset_dir}")
        return

    # Build per-run curves: mean over clients per iteration, for each (server_net_type_value, path, seed)
    run_rows = []
    for (sn, path, seed), d in df_all.groupby(["server_net_type_value", "_path", "seed"], dropna=False):
        s = _aggregate_run_mean_over_clients(d[["iteration", "client_id", "accuracy"]].copy())
        if s.empty:
            continue
        run_rows.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "run_value": _maybe_scale_to_percent(s.values.astype(float)),
            "server_net_type_value": str(sn),
            "seed": seed,
            "_path": path,
        }))

    if not run_rows:
        print(f"[WARN] No per-run curves produced for {figset_dir}")
        return

    df_runs = pd.concat(run_rows, ignore_index=True)

    # Final CSV only
    df_final = _final_stats_for_diff_server_nets_from_runs(df_runs)
    _save_df_csv(df_final, out_root / "diff_server_nets_final.csv")

    # Per-iteration mean curve (for plotting)
    df_iter = (
        df_runs.groupby(["server_net_type_value", "iter"], dropna=False)["run_value"]
              .mean()
              .reset_index()
              .rename(columns={"run_value": "mean"})
              .sort_values(["server_net_type_value", "iter"])
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    keys = sorted(df_iter["server_net_type_value"].unique().tolist())

    cmap = plt.get_cmap("tab10")
    for i, sn in enumerate(keys):
        d = df_iter[df_iter["server_net_type_value"] == sn].sort_values("iter")
        override = _color_override_server_net(sn)
        ax.plot(
            d["iter"], d["mean"],
            linewidth=3.0, marker=None,
            label=str(sn),
            color=(override if override is not None else cmap(i % 10)),
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Top-1 Accuracy")
    ax.grid(False)
    _force_x_axes_0_to_9(fig)

    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(6, len(h)),
                   frameon=False, bbox_to_anchor=(0.5, 0.96),
                   fontsize=LEGEND_FONT_SIZE)

    out_pdf = out_root / "diff_server_nets.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

# --------------------- Combined: MAPL lambda + temp (two panel) + FINAL CSVs only ---------------------

def _normalize_accuracy_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    d = df.copy()

    mask_cifar = d["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        d = d[mask_cifar].copy()

    d = _apply_measure_filter_for_comparison(d)
    if d is None or d.empty:
        return d

    is_mapl = d["algorithm_display"].astype(str).str.lower().str.contains("mapl")
    d = d[is_mapl].copy()
    if d.empty:
        return d

    if "iteration" in d.columns:
        d["iteration"] = pd.to_numeric(d["iteration"], errors="coerce")
        d = d.dropna(subset=["iteration"])
        d["iteration"] = d["iteration"].astype(int)

    if "lambda_consistency" in d.columns:
        d["lambda_consistency"] = pd.to_numeric(d["lambda_consistency"], errors="coerce")

    if "distill_temperature" in d.columns:
        d["distill_temperature"] = pd.to_numeric(d["distill_temperature"], errors="coerce")

    if "server_input_tech_name" in d.columns:
        d["server_input_tech_name"] = d["server_input_tech_name"].astype(str)
        d = d[d["server_input_tech_name"].str.lower().ne("none")].copy()

    return d

def figure_mapl_lambda_and_temp_two_panel(lambda_dir: Path, temp_dir: Path, out_root: Path, *, inspect: bool):
    if not lambda_dir.exists():
        print(f"[SKIP] {lambda_dir} (missing)")
        return
    if not temp_dir.exists():
        print(f"[SKIP] {temp_dir} (missing)")
        return

    df_lam = _normalize_accuracy_df(_load_any_jsons_under(lambda_dir, inspect=inspect))
    if df_lam is None or df_lam.empty:
        print(f"[WARN] No MAPL lambda rows under {lambda_dir}")
        return

    df_tmp = _normalize_accuracy_df(_load_any_jsons_under(temp_dir, inspect=inspect))
    if df_tmp is None or df_tmp.empty:
        print(f"[WARN] No MAPL temp rows under {temp_dir}")
        return

    # Build per-run curves (mean over clients) for each setting
    runs_lam = (
        df_lam.groupby(["lambda_consistency","_path","seed","iteration"], dropna=False)["accuracy"]
              .mean()
              .reset_index()
              .rename(columns={"iteration":"iter", "accuracy":"run_value"})
    )
    runs_lam["run_value"] = _maybe_scale_to_percent(runs_lam["run_value"].to_numpy(dtype=float))

    runs_tmp = (
        df_tmp.groupby(["server_input_tech_name","distill_temperature","_path","seed","iteration"], dropna=False)["accuracy"]
              .mean()
              .reset_index()
              .rename(columns={"iteration":"iter", "accuracy":"run_value"})
    )
    runs_tmp["run_value"] = _maybe_scale_to_percent(runs_tmp["run_value"].to_numpy(dtype=float))

    # Final CSVs only
    lam_final = _final_stats_for_mapl_lambda_from_runs(runs_lam, final_iter=FINAL_ITER)
    _save_df_csv(lam_final, out_root / "mapl_lambda_and_temp_lambda_final.csv")

    tmp_final = _final_stats_for_temp_serverinput_from_runs(runs_tmp, final_iter=FINAL_ITER)
    _save_df_csv(tmp_final, out_root / "mapl_lambda_and_temp_temp_final.csv")

    # Per-iteration means for plotting
    lam_iter = (
        runs_lam.groupby(["lambda_consistency","iter"], dropna=False)["run_value"]
                .mean().reset_index().rename(columns={"run_value":"mean"})
                .sort_values(["lambda_consistency","iter"])
    )
    tmp_iter = (
        runs_tmp.groupby(["server_input_tech_name","distill_temperature","iter"], dropna=False)["run_value"]
                .mean().reset_index().rename(columns={"run_value":"mean"})
                .sort_values(["distill_temperature","server_input_tech_name","iter"])
    )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    for lam, g in lam_iter.groupby("lambda_consistency"):
        g = g.sort_values("iter")
        ax_l.plot(g["iter"], g["mean"], label=f"λ={lam:g}", linewidth=3.0)
    ax_l.set_title("MAPL – Consistency Weight λ")
    ax_l.set_xlabel("Iteration")
    ax_l.set_ylabel("Top-1 Accuracy")
    ax_l.grid(False)
    ax_l.legend(frameon=False)

    for (tech, temp), g in tmp_iter.groupby(["server_input_tech_name", "distill_temperature"]):
        g = g.sort_values("iter")
        ax_r.plot(g["iter"], g["mean"], label=f"{tech}, T={temp:g}", linewidth=3.0)
    ax_r.set_title("Distillation Temperature × Server Input")
    ax_r.set_xlabel("Iteration")
    ax_r.set_ylabel("Top-1 Accuracy")
    ax_r.grid(False)
    ax_r.legend(frameon=False, fontsize=8)

    _force_x_axes_0_to_9(fig)

    out_pdf = out_root / "mapl_lambda_and_temp.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

# --------------------- Combined: client_scale + global_data_size (two panel) + FINAL CSVs only ---------------------

def figure_client_scale_and_global_data_size_two_panel(
    client_scale_dir: Path,
    global_data_dir: Path,
    out_root: Path,
    *,
    inspect: bool,
    use_server_measure: bool = False,
):
    if not client_scale_dir.exists():
        print(f"[SKIP] {client_scale_dir} (missing)")
        return
    if not global_data_dir.exists():
        print(f"[SKIP] {global_data_dir} (missing)")
        return

    # LEFT: client_scale runs
    df_all = _load_any_jsons_under(client_scale_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {client_scale_dir}")
        return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {client_scale_dir}")
        return

    if "num_clients" not in df_all.columns or df_all["num_clients"].isna().all():
        print(f"[WARN] No summary[num_clients] detected in {client_scale_dir}")
        return

    df_all = df_all.copy()
    df_all["num_clients"] = pd.to_numeric(df_all["num_clients"], errors="coerce")
    df_all = df_all.dropna(subset=["num_clients"])

    run_rows_cs = []
    for (path, nc, seed), d in df_all.groupby(["_path", "num_clients", "seed"], dropna=False):
        s = _aggregate_run_mean_over_clients(d[["iteration","client_id","accuracy"]].copy())
        if s.empty:
            continue
        run_rows_cs.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "run_value": _maybe_scale_to_percent(s.values.astype(float)),
            "num_clients": int(nc),
            "seed": seed,
            "_path": path,
        }))
    if not run_rows_cs:
        print(f"[WARN] No per-run curves produced for {client_scale_dir}")
        return

    runs_cs = pd.concat(run_rows_cs, ignore_index=True)
    df_final_cs = _final_stats_for_client_scale_from_iter(runs_cs)

    _save_df_csv(df_final_cs, out_root / "client_scale_and_global_data_size_client_scale_final.csv")

    df_iter_cs = (
        runs_cs.groupby(["num_clients","iter"], dropna=False)["run_value"]
               .mean().reset_index().rename(columns={"run_value":"mean"})
               .sort_values(["num_clients","iter"])
    )

    # RIGHT: global_data_size curves
    curve_dirs = [d for d in sorted(global_data_dir.iterdir()) if d.is_dir()]
    if not curve_dirs:
        print(f"[WARN] No curve folders under {global_data_dir}")
        return

    ratio_curves: List[Tuple[str, pd.DataFrame]] = []
    for cd in curve_dirs:
        df = load_rows_from_dir(cd, inspect=inspect, alg_hint=None)
        if df.empty:
            continue
        mask_cifar_g = df["dataset"].astype(str).str.lower() == "cifar100"
        if mask_cifar_g.any():
            df = df[mask_cifar_g]
        if df.empty:
            continue

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

        r = df.get("server_data_ratio", pd.Series([], dtype=float)).dropna()
        if len(r) > 0:
            ratio = float(r.iloc[0])
            label = f"{ratio:g}"
        else:
            ratio, _ = find_server_data_ratio({}, fallback_from=cd.name)
            label = f"{ratio:g}" if ratio is not None else cd.name

        g = (
            df.groupby(["iteration"], dropna=False)["accuracy"]
              .mean()
              .reset_index()
              .rename(columns={"accuracy": "avg_accuracy"})
              .sort_values(["iteration"])
        )
        if not g.empty:
            ratio_curves.append((label, g))

    if not ratio_curves:
        print(f"[WARN] No usable rows with server_data_ratio in {global_data_dir}")
        return

    def _ratio_key(lbl: str):
        try:
            return float(lbl)
        except Exception:
            return float("inf")
    ratio_curves.sort(key=lambda t: _ratio_key(t[0]))

    # Final CSV for global data size (at iteration==FINAL_ITER if exists, else max iteration per curve)
    final_gd_rows = []
    for lab, g in ratio_curves:
        g_it = pd.to_numeric(g["iteration"], errors="coerce").dropna()
        if g_it.empty:
            continue
        if (g_it == FINAL_ITER).any():
            it = FINAL_ITER
        else:
            it = int(g_it.max())
        row = g[g["iteration"] == it]
        if row.empty:
            continue
        final_gd_rows.append({
            "server_data_ratio": lab,
            "iteration": int(it),
            "mean": float(row["avg_accuracy"].iloc[0]),
        })
    df_global_final = pd.DataFrame(final_gd_rows, columns=["server_data_ratio","iteration","mean"])
    _save_df_csv(df_global_final, out_root / "client_scale_and_global_data_size_global_data_final.csv")

    # Plot combined
    fig, (ax_cs, ax_gd) = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=False)

    for nc in sorted(df_iter_cs["num_clients"].unique().tolist()):
        d = df_iter_cs[df_iter_cs["num_clients"] == nc].sort_values("iter")
        ax_cs.plot(d["iter"], d["mean"], linewidth=3.0, label=f"{int(nc)} clients")

    ax_cs.set_xlabel("Iteration")
    ax_cs.set_ylabel("Top-1 Accuracy")
    ax_cs.grid(False)
    h_cs, l_cs = ax_cs.get_legend_handles_labels()
    if h_cs:
        ax_cs.legend(h_cs, l_cs, loc="center left", bbox_to_anchor=(1.02, 0.5),
                     frameon=False, fontsize=LEGEND_FONT_SIZE, title="# Clients")

    cmap = plt.get_cmap("tab10")
    for i, (lab, g) in enumerate(ratio_curves):
        ax_gd.plot(g["iteration"], g["avg_accuracy"], linewidth=3.0, label=lab, color=cmap(i % 10))

    ax_gd.set_xlabel("Iteration")
    ax_gd.set_ylabel("Top-1 Accuracy")
    ax_gd.grid(False)
    h_gd, l_gd = ax_gd.get_legend_handles_labels()
    if h_gd:
        ax_gd.legend(h_gd, l_gd, loc="center left", bbox_to_anchor=(1.02, 0.5),
                     frameon=False, fontsize=LEGEND_FONT_SIZE, title="Server Data Ratio")

    for ax in (ax_cs, ax_gd):
        ax.set_ylim(19, 50)

    ax_cs.text(0.02, 0.98, "(a)", transform=ax_cs.transAxes, ha="left", va="top",
               fontsize=LEGEND_FONT_SIZE, fontweight="bold")
    ax_gd.text(0.02, 0.98, "(b)", transform=ax_gd.transAxes, ha="left", va="top",
               fontsize=LEGEND_FONT_SIZE, fontweight="bold")

    _force_x_axes_0_to_9(fig)
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

    suffix = "server" if use_server_measure else "client"
    out_pdf = out_root / f"client_scale_and_global_data_size_{suffix}.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")



def _final_stats_for_diff_server_nets_from_runs(df_runs: pd.DataFrame, *, final_iter: int = FINAL_ITER) -> pd.DataFrame:
    d = df_runs.copy()
    d["iter"] = pd.to_numeric(d["iter"], errors="coerce")
    d = d[d["iter"] == final_iter].copy()
    if d.empty:
        return pd.DataFrame(columns=["server_net_type_value","iteration","mean","std","sem","n"])

    out = []
    for sn, g in d.groupby("server_net_type_value"):
        vals = pd.to_numeric(g["run_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        n = int(vals.size)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        out.append({
            "server_net_type_value": str(sn),
            "iteration": int(final_iter),
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
        })

    return pd.DataFrame(out, columns=["server_net_type_value","iteration","mean","std","sem","n"])


def _normalize_diff_server_nets_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep CIFAR100 if present, apply MAPL policy filter, then keep MAPL rows,
    ensure server_net_type_value exists and is usable.
    """
    if df is None or df.empty:
        return df

    d = df.copy()

    mask_cifar = d["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        d = d[mask_cifar].copy()

    d = _apply_measure_filter_for_comparison(d)
    if d is None or d.empty:
        return d

    # only MAPL (since you're varying server nets for MAPL)
    is_mapl = d["algorithm_display"].astype(str).str.lower().str.contains("mapl")
    d = d[is_mapl].copy()
    if d.empty:
        return d

    if "server_net_type_value" not in d.columns:
        d["server_net_type_value"] = None

    # clean missing/None
    sn = d["server_net_type_value"].astype(str)
    bad = sn.isna() | sn.str.lower().isin(["none", "nan", ""])
    d = d[~bad].copy()

    if "iteration" in d.columns:
        d["iteration"] = pd.to_numeric(d["iteration"], errors="coerce")
        d = d.dropna(subset=["iteration"])
        d["iteration"] = d["iteration"].astype(int)

    return d

def figure_client_scale_and_global_data_size_three_panel(
    client_scale_dir: Path,
    global_data_dir: Path,
    diff_server_nets_dir: Path,
    out_root: Path,
    *,
    inspect: bool,
    use_server_measure: bool = False,
):
    if not client_scale_dir.exists():
        print(f"[SKIP] {client_scale_dir} (missing)")
        return
    if not global_data_dir.exists():
        print(f"[SKIP] {global_data_dir} (missing)")
        return
    if not diff_server_nets_dir.exists():
        print(f"[SKIP] {diff_server_nets_dir} (missing)")
        return

    # ============================================================
    # (a) LEFT: client_scale runs
    # ============================================================
    df_all = _load_any_jsons_under(client_scale_dir, inspect=inspect)
    if df_all.empty:
        print(f"[WARN] No data under {client_scale_dir}")
        return

    mask_cifar = df_all["dataset"].astype(str).str.lower() == "cifar100"
    if mask_cifar.any():
        df_all = df_all[mask_cifar]

    df_all = _apply_measure_filter_for_comparison(df_all)
    if df_all.empty:
        print(f"[WARN] No usable rows (policy-filtered) under {client_scale_dir}")
        return

    if "num_clients" not in df_all.columns or df_all["num_clients"].isna().all():
        print(f"[WARN] No summary[num_clients] detected in {client_scale_dir}")
        return

    df_all = df_all.copy()
    df_all["num_clients"] = pd.to_numeric(df_all["num_clients"], errors="coerce")
    df_all = df_all.dropna(subset=["num_clients"])

    run_rows_cs = []
    for (path, nc, seed), d in df_all.groupby(["_path", "num_clients", "seed"], dropna=False):
        s = _aggregate_run_mean_over_clients(d[["iteration", "client_id", "accuracy"]].copy())
        if s.empty:
            continue
        run_rows_cs.append(pd.DataFrame({
            "iter": s.index.astype(int),
            "run_value": _maybe_scale_to_percent(s.values.astype(float)),
            "num_clients": int(nc),
            "seed": seed,
            "_path": path,
        }))
    if not run_rows_cs:
        print(f"[WARN] No per-run curves produced for {client_scale_dir}")
        return

    runs_cs = pd.concat(run_rows_cs, ignore_index=True)
    df_final_cs = _final_stats_for_client_scale_from_iter(runs_cs)
    _save_df_csv(df_final_cs, out_root / "client_scale_and_global_data_size_client_scale_final.csv")

    df_iter_cs = (
        runs_cs.groupby(["num_clients", "iter"], dropna=False)["run_value"]
               .mean()
               .reset_index()
               .rename(columns={"run_value": "mean"})
               .sort_values(["num_clients", "iter"])
    )

    # ============================================================
    # (b) MIDDLE: global_data_size curves
    # ============================================================
    curve_dirs = [d for d in sorted(global_data_dir.iterdir()) if d.is_dir()]
    if not curve_dirs:
        print(f"[WARN] No curve folders under {global_data_dir}")
        return

    ratio_curves: List[Tuple[str, pd.DataFrame]] = []
    for cd in curve_dirs:
        df = load_rows_from_dir(cd, inspect=inspect, alg_hint=None)
        if df.empty:
            continue

        mask_cifar_g = df["dataset"].astype(str).str.lower() == "cifar100"
        if mask_cifar_g.any():
            df = df[mask_cifar_g]
        if df.empty:
            continue

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

        r = df.get("server_data_ratio", pd.Series([], dtype=float)).dropna()
        if len(r) > 0:
            ratio = float(r.iloc[0])
            label = f"{ratio:g}"
        else:
            ratio, _ = find_server_data_ratio({}, fallback_from=cd.name)
            label = f"{ratio:g}" if ratio is not None else cd.name

        g = (
            df.groupby(["iteration"], dropna=False)["accuracy"]
              .mean()
              .reset_index()
              .rename(columns={"accuracy": "avg_accuracy"})
              .sort_values(["iteration"])
        )
        if not g.empty:
            ratio_curves.append((label, g))

    if not ratio_curves:
        print(f"[WARN] No usable rows with server_data_ratio in {global_data_dir}")
        return

    def _ratio_key(lbl: str):
        try:
            return float(lbl)
        except Exception:
            return float("inf")
    ratio_curves.sort(key=lambda t: _ratio_key(t[0]))

    final_gd_rows = []
    for lab, g in ratio_curves:
        g_it = pd.to_numeric(g["iteration"], errors="coerce").dropna()
        if g_it.empty:
            continue
        it = FINAL_ITER if (g_it == FINAL_ITER).any() else int(g_it.max())
        row = g[g["iteration"] == it]
        if row.empty:
            continue
        final_gd_rows.append({
            "server_data_ratio": lab,
            "iteration": int(it),
            "mean": float(row["avg_accuracy"].iloc[0]),
        })
    df_global_final = pd.DataFrame(final_gd_rows, columns=["server_data_ratio", "iteration", "mean"])
    _save_df_csv(df_global_final, out_root / "client_scale_and_global_data_size_global_data_final.csv")

    # ============================================================
    # (c) RIGHT: diff_server_nets (new)
    # ============================================================
    df_srv = _normalize_diff_server_nets_df(_load_any_jsons_under(diff_server_nets_dir, inspect=inspect))
    if df_srv is None or df_srv.empty:
        print(f"[WARN] No usable MAPL diff_server_nets rows under {diff_server_nets_dir}")
        return

    runs_srv = (
        df_srv.groupby(["server_net_type_value", "_path", "seed", "iteration"], dropna=False)["accuracy"]
              .mean()
              .reset_index()
              .rename(columns={"iteration": "iter", "accuracy": "run_value"})
    )
    runs_srv["run_value"] = _maybe_scale_to_percent(runs_srv["run_value"].to_numpy(dtype=float))

    srv_final = _final_stats_for_diff_server_nets_from_runs(runs_srv, final_iter=FINAL_ITER)
    _save_df_csv(srv_final, out_root / "client_scale_and_global_data_size_diff_server_nets_final.csv")

    srv_iter = (
        runs_srv.groupby(["server_net_type_value", "iter"], dropna=False)["run_value"]
                .mean()
                .reset_index()
                .rename(columns={"run_value": "mean"})
                .sort_values(["server_net_type_value", "iter"])
    )

    # ============================================================
    # Plot 3-panel combined
    # ============================================================
    fig, (ax_cs, ax_gd, ax_srv) = plt.subplots(1, 3, figsize=(18.0, 4.8), constrained_layout=False)

    # (a) client_scale
    for nc in sorted(df_iter_cs["num_clients"].unique().tolist()):
        d = df_iter_cs[df_iter_cs["num_clients"] == nc].sort_values("iter")
        ax_cs.plot(d["iter"], d["mean"], linewidth=3.0, label=f"{int(nc)} clients")
    ax_cs.set_xlabel("Iteration")
    ax_cs.set_ylabel("Top-1 Accuracy")
    ax_cs.grid(False)
    h_cs, l_cs = ax_cs.get_legend_handles_labels()
    if h_cs:
        ax_cs.legend(h_cs, l_cs, loc="center left", bbox_to_anchor=(1.02, 0.5),
                     frameon=False, fontsize=LEGEND_FONT_SIZE, title="# Clients")

    # (b) global_data_size
    cmap = plt.get_cmap("tab10")
    for i, (lab, g) in enumerate(ratio_curves):
        override = _color_override_global_data_ratio(lab)
        ax_gd.plot(
            g["iteration"], g["avg_accuracy"],
            linewidth=3.0,
            label=lab,
            color=(override if override is not None else cmap(i % 10)),
        )

    ax_gd.set_xlabel("Iteration")
    ax_gd.set_ylabel("Top-1 Accuracy")
    ax_gd.grid(False)
    h_gd, l_gd = ax_gd.get_legend_handles_labels()
    if h_gd:
        ax_gd.legend(h_gd, l_gd, loc="center left", bbox_to_anchor=(1.02, 0.5),
                     frameon=False, fontsize=LEGEND_FONT_SIZE, title="Server Data Ratio")

    # (c) diff_server_nets
    keys = sorted(srv_iter["server_net_type_value"].astype(str).unique().tolist())
    for i, sn in enumerate(keys):
        d = srv_iter[srv_iter["server_net_type_value"].astype(str) == str(sn)].sort_values("iter")
        override = _color_override_server_net(sn)
        ax_srv.plot(
            d["iter"], d["mean"],
            linewidth=3.0,
            label=str(sn),
            color=(override if override is not None else cmap(i % 10)),
        )
    ax_srv.set_xlabel("Iteration")
    ax_srv.set_ylabel("Top-1 Accuracy")
    ax_srv.grid(False)
    h_s, l_s = ax_srv.get_legend_handles_labels()
    if h_s:
        ax_srv.legend(h_s, l_s, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=False, fontsize=LEGEND_FONT_SIZE, title="Server Net")

    # shared cosmetics
    for ax in (ax_cs, ax_gd, ax_srv):
        ax.set_ylim(19, 50)

    ax_cs.text(0.02, 0.98, "(a)", transform=ax_cs.transAxes, ha="left", va="top",
               fontsize=LEGEND_FONT_SIZE, fontweight="bold")
    ax_gd.text(0.02, 0.98, "(b)", transform=ax_gd.transAxes, ha="left", va="top",
               fontsize=LEGEND_FONT_SIZE, fontweight="bold")
    ax_srv.text(0.02, 0.98, "(c)", transform=ax_srv.transAxes, ha="left", va="top",
                fontsize=LEGEND_FONT_SIZE, fontweight="bold")

    _force_x_axes_0_to_9(fig)
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])  # space for legends

    suffix = "server" if use_server_measure else "client"
    out_pdf = out_root / f"client_scale_and_global_data_size_{suffix}.pdf"
    _savefig_longpath(fig, out_pdf)
    plt.close(fig)
    print(f"[OK] Saved: {out_pdf}")

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"), help="Root results folder (contains figure-set folders)")
    ap.add_argument("--figdir", type=Path, default=Path("figures"), help="Where to save PDFs/CSVs (no subfolders created)")
    ap.add_argument("--inspect", action="store_true", help="Print detected paths/keys while loading")
    ap.add_argument("--global_data_size_folder", type=str, default="global_data_size",
                    help="Folder under --root where each immediate subfolder represents one curve/ratio.")
    ap.add_argument("--client_scale_folder", type=str, default="client_scale",
                    help="Folder under --root containing runs with different summary[num_clients].")
    ap.add_argument("--diff_server_nets_folder", type=str, default="diff_server_nets",
                    help="Folder under --root containing MAPL runs with different server_net_type_value.")

    args = ap.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)

    # Global-data-size: SERVER-only (+ PDF.csv via _save_stats_csv)
    figure_global_data_size_oneplot(
        args.root / args.global_data_size_folder,
        args.figdir,
        inspect=args.inspect,
        use_server_measure=True
    )

    # Client-scale (+ final CSV only)
    figure_client_scale_oneplot(
        args.root / args.client_scale_folder,
        args.figdir,
        inspect=args.inspect
    )

    # Comparison figures (+ PDF.csv via _save_stats_csv)
    figure_diff_clients_nets(args.root / "diff_clients_nets", args.figdir, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "diff_benchmarks_05", args.figdir, alpha_value=5, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "diff_benchmarks_100", args.figdir, alpha_value=100, inspect=args.inspect)
    figure_diff_benchmarks(args.root / "diff_benchmarks_1", args.figdir, alpha_value=1, inspect=args.inspect)

    figure_by_client_net_type_value(args.root / "same_client_nets", args.figdir, inspect=args.inspect)

    # temp_serverinput: final CSV only
    figure_temp_serverinput_oneplot(
        args.root / "temp_serverinput",
        args.figdir,
        inspect=args.inspect,
    )

    # mapl_lambda: final CSV only
    figure_mapl_lambda_oneplot(
        args.root / "mapl_lambda",
        args.figdir,
        inspect=args.inspect,
    )

    # NEW: diff_server_nets (MAPL server only): final CSV only
    figure_diff_server_nets_oneplot(
        args.root / args.diff_server_nets_folder,
        args.figdir,
        inspect=args.inspect,
    )

    # combined MAPL lambda+temp: final CSVs only
    figure_mapl_lambda_and_temp_two_panel(
        lambda_dir=args.root / "mapl_lambda",
        temp_dir=args.root / "temp_serverinput",
        out_root=args.figdir,
        inspect=args.inspect,
    )

    # Combined client_scale + global_data_size: final CSVs only
    figure_client_scale_and_global_data_size_three_panel(
        client_scale_dir=args.root / "client_scale",
        global_data_dir=args.root / "global_data_size",
        diff_server_nets_dir=args.root / "diff_server_nets",
        out_root=args.figdir,
        inspect=args.inspect,
        use_server_measure=True,
    )

if __name__ == "__main__":
    main()
