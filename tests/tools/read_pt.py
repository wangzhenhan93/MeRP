#!/usr/bin/env python3
"""
compare_prompts.py

Compare two PyTorch .pt / checkpoint files that likely contain prompt encoders or state_dicts.
Produces a human-readable report and a JSON summary.

Usage:
   python tools/read_pt.py prompts/bur-en_step500.pt prompts/prompt_enc_bur_en.pt --out tools/report.json


Notes:
 - The script tries to load with torch.load(..., weights_only=True) if supported to reduce pickle attack surface,
   but falls back to plain torch.load if not available.
 - It extracts dict-like "state_dict" candidates from common keys:
     'encoder_state_dict', 'state_dict', 'model_state_dict', 'prompt', etc.
 - If files are not dicts but raw state_dicts, it will attempt to treat them directly.
"""
import argparse
import json
import math
import sys
from collections import OrderedDict

import torch
import numpy as np


COMMON_STATE_KEYS = ("encoder_state_dict", "state_dict", "model_state_dict", "model", "encoder", "prompt")


def safe_torch_load(path: str):
    """
    Try to load with weights_only=True if available (newer torch), else fallback.
    Returns loaded object.
    """
    try:
        # weights_only was introduced in newer torch versions
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # older torch doesn't accept weights_only
        return torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")


def find_best_state_dict(obj):
    """
    Given loaded object (maybe dict or state_dict), return a tuple (state_dict, meta)
    where state_dict is a dict[str->Tensor] and meta contains config/metadata keys if any.
    """
    if isinstance(obj, dict):
        # if this dict *is* the state_dict (values are tensors or numpy)
        # Heuristic: many keys like 'xyz.weight' present and values are tensors
        example_values = list(obj.values())[:5]
        if all(isinstance(v, (torch.Tensor, np.ndarray)) for v in example_values):
            return obj, {k: obj[k] for k in ("config", "metadata") if k in obj}  # treat whole dict as state_dict
        # else, search common keys
        for k in COMMON_STATE_KEYS:
            if k in obj and isinstance(obj[k], dict):
                return obj[k], {kk: obj.get(kk) for kk in ("config", "metadata", "created_at", "step", "epoch") if kk in obj}
        # fallback: try to find any dict-of-tensors value
        for k, v in obj.items():
            if isinstance(v, dict):
                # check whether dict looks like state_dict
                vals = list(v.values())[:5]
                if len(vals) > 0 and all(isinstance(x, (torch.Tensor, np.ndarray)) for x in vals):
                    return v, {kk: obj.get(kk) for kk in ("config", "metadata", "created_at", "step", "epoch") if kk in obj}
        # no internal state_dict found -> return none
        return None, {kk: obj.get(kk) for kk in ("config", "metadata", "created_at", "step", "epoch") if kk in obj}
    else:
        # not a dict - cannot extract
        return None, {}


def summarize_state(keys_to_tensors: dict):
    summary = {}
    total_params = 0
    for k, v in keys_to_tensors.items():
        if isinstance(v, np.ndarray):
            arr = v
            t = torch.from_numpy(arr)
        elif isinstance(v, torch.Tensor):
            t = v.detach().cpu()
        else:
            # skip non-tensor entries
            continue
        n = t.numel()
        total_params += n
        # small stats
        with torch.no_grad():
            s = t.float()
            mean = float(s.mean().item()) if n > 0 else 0.0
            std = float(s.std().item()) if n > 0 else 0.0
            mn = float(s.min().item()) if n > 0 else 0.0
            mx = float(s.max().item()) if n > 0 else 0.0
            l2 = float(torch.norm(s).item()) if n > 0 else 0.0
        summary[k] = {"shape": tuple(t.shape), "numel": n, "mean": mean, "std": std, "min": mn, "max": mx, "l2": l2}
    return summary, total_params


def compare_states(a_state: dict, b_state: dict, tol: float = 1e-6, topk: int = 20):
    """
    Compare two state_dict-like mappings.
    Returns a dict with per-key comparison, plus summary statistics.
    """
    a_keys = set(k for k in a_state.keys() if isinstance(a_state[k], (torch.Tensor, np.ndarray)))
    b_keys = set(k for k in b_state.keys() if isinstance(b_state[k], (torch.Tensor, np.ndarray)))

    common = sorted(list(a_keys & b_keys))
    only_a = sorted(list(a_keys - b_keys))
    only_b = sorted(list(b_keys - a_keys))

    per_key = OrderedDict()
    changed_count = 0
    total_numel = 0
    total_changed_elements = 0
    sum_rel_diff = 0.0
    sum_abs_diff_l2 = 0.0

    top_diffs = []  # (key, l2diff, reldiff, numel)

    for k in common:
        a = a_state[k]
        b = b_state[k]
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b)
        a = a.detach().cpu().float()
        b = b.detach().cpu().float()
        if tuple(a.shape) != tuple(b.shape):
            per_key[k] = {"shape_a": tuple(a.shape), "shape_b": tuple(b.shape), "note": "shape_mismatch"}
            continue
        n = a.numel()
        total_numel += n
        diff = (a - b)
        l2_a = float(torch.norm(a).item())
        l2_b = float(torch.norm(b).item())
        l2_diff = float(torch.norm(diff).item())
        # relative difference: normalized by l2 of a (or b) to avoid divide by zero
        denom = max(l2_a, l2_b, 1e-12)
        rel = l2_diff / denom
        # element-wise changed count (abs diff > tol)
        changed_elems = int((diff.abs() > float(tol)).sum().item())
        total_changed_elements += changed_elems
        if changed_elems > 0:
            changed_count += 1
        sum_rel_diff += rel
        sum_abs_diff_l2 += l2_diff
        per_key[k] = {
            "shape": tuple(a.shape),
            "numel": n,
            "l2_a": l2_a,
            "l2_b": l2_b,
            "l2_diff": l2_diff,
            "rel_l2_diff": rel,
            "changed_elements": changed_elems,
            "changed_ratio": changed_elems / n if n > 0 else 0.0,
        }
        top_diffs.append((k, l2_diff, rel, n))

    # sort top diffs by absolute l2_diff desc
    top_diffs = sorted(top_diffs, key=lambda x: x[1], reverse=True)[:topk]

    summary = {
        "n_common": len(common),
        "n_only_in_a": len(only_a),
        "n_only_in_b": len(only_b),
        "n_tensors_changed": changed_count,
        "total_tensors_a": len(a_keys),
        "total_tensors_b": len(b_keys),
        "total_numel_common": int(total_numel),
        "total_changed_elements": int(total_changed_elements),
        "percent_elements_changed": float(total_changed_elements / max(total_numel, 1) * 100.0) if total_numel > 0 else 0.0,
        "avg_rel_l2_diff": float(sum_rel_diff / max(len(common), 1)),
        "sum_abs_l2_diff": float(sum_abs_diff_l2),
        "top_diffs": [{"key": k, "l2_diff": l2, "rel_l2": rel, "numel": n} for (k, l2, rel, n) in top_diffs],
        "only_in_a": only_a,
        "only_in_b": only_b,
    }
    return per_key, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left", help="left .pt file (e.g. older)")
    ap.add_argument("right", help="right .pt file (e.g. newer)")
    ap.add_argument("--out", help="json output file", default="compare_report.json")
    ap.add_argument("--topk", type=int, default=20, help="how many top differing tensors to show")
    ap.add_argument("--tol", type=float, default=1e-6, help="element-wise difference tolerance")
    args = ap.parse_args()

    print(f"Loading left: {args.left}")
    left_obj = safe_torch_load(args.left)
    print(f"Loading right: {args.right}")
    right_obj = safe_torch_load(args.right)

    left_state, left_meta = find_best_state_dict(left_obj)
    right_state, right_meta = find_best_state_dict(right_obj)

    report = {"left_path": args.left, "right_path": args.right, "left_meta": left_meta, "right_meta": right_meta}

    if left_state is None:
        print("Warning: could not detect a state-dict-like mapping in left file.")
        left_state = {}
    if right_state is None:
        print("Warning: could not detect a state-dict-like mapping in right file.")
        right_state = {}

    print("Summarizing left file tensors...")
    left_summary, left_total = summarize_state(left_state)
    print("Summarizing right file tensors...")
    right_summary, right_total = summarize_state(right_state)

    report["left_summary"] = {"n_tensors": len(left_summary), "total_params": int(left_total)}
    report["right_summary"] = {"n_tensors": len(right_summary), "total_params": int(right_total)}

    # Show some metadata hints (created_at/step/config keys)
    meta_info = {}
    meta_info["left_meta_keys"] = list(left_meta.keys())
    meta_info["right_meta_keys"] = list(right_meta.keys())
    for k in ("created_at", "step", "epoch", "config", "metadata"):
        if k in left_meta:
            meta_info["left_"+k] = left_meta[k]
        if k in right_meta:
            meta_info["right_"+k] = right_meta[k]
    report["meta_info"] = meta_info

    print("Comparing states ... (this may take a few seconds for many tensors)")
    per_key, summary = compare_states(left_state, right_state, tol=args.tol, topk=args.topk)
    report["per_tensor"] = per_key
    report["summary"] = summary

    # Small human readable prints:
    print("\n=== Quick summary ===")
    print(f"Left tensors: {len(left_summary)}  total params: {left_total}")
    print(f"Right tensors: {len(right_summary)} total params: {right_total}")
    print(f"Common tensors: {summary['n_common']}")
    print(f"Tensors only in left: {len(summary['only_in_a'])}, only in right: {len(summary['only_in_b'])}")
    print(f"Tensors with any changed elements: {summary['n_tensors_changed']}")
    print(f"Total common elements: {summary['total_numel_common']:,}")
    print(f"Changed elements: {summary['total_changed_elements']:,} ({summary['percent_elements_changed']:.6f}%)")
    print(f"Avg relative L2 diff across tensors: {summary['avg_rel_l2_diff']:.6e}")
    print(f"Sum absolute L2 diff: {summary['sum_abs_l2_diff']:.6e}")

    print("\nTop differing tensors (by L2 diff):")
    for it in summary["top_diffs"]:
        print(f" - {it['key']}: l2_diff={it['l2_diff']:.6e}  rel_l2={it['rel_l2']:.6e}  numel={it['numel']}")

    # Save full report JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report written to {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
