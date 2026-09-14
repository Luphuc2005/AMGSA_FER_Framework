#!/usr/bin/env python3
"""Contract check and precise semantic diff verification between new candidate and 91% baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def compute_dict_diff(d1: dict, d2: dict, prefix: str = "") -> list:
    """Recursively find differences between two dictionaries."""
    diffs = []
    all_keys = sorted(set(d1.keys()) | set(d2.keys()))
    for k in all_keys:
        full_key = f"{prefix}.{k}" if prefix else k
        if k not in d1:
            diffs.append((full_key, "<absent>", d2[k]))
        elif k not in d2:
            diffs.append((full_key, d1[k], "<absent>"))
        elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
            diffs.extend(compute_dict_diff(d1[k], d2[k], prefix=full_key))
        else:
            val1 = d1[k]
            val2 = d2[k]
            # Normalize float comparisons
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(float(val1) - float(val2)) > 1e-6:
                    diffs.append((full_key, val1, val2))
            elif val1 != val2:
                diffs.append((full_key, val1, val2))
    return diffs


def print_and_verify_diff(base_path: Path, cand_path: Path):
    with base_path.open("r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)
    with cand_path.open("r", encoding="utf-8") as f:
        cand_raw = yaml.safe_load(f)

    diffs = compute_dict_diff(base_raw, cand_raw)

    print("=" * 90)
    print(f"DIFF INSPECTION: Baseline 91% vs New Candidate")
    print(f"  Baseline:  {base_path.name}")
    print(f"  Candidate: {cand_path.name}")
    print("=" * 90)
    print(f"{'Key / Parameter':<42} | {'Baseline 91%':<22} | {'New Candidate':<22}")
    print("-" * 90)

    # Permitted intentional changes
    intentional_prefixes = {
        "source.note",
        "model.name",
        "paths.output_dir",
        # 1. Direct Global + Regional Fusion
        "model.use_dynamic_part_attention",
        "model.dynamic_part_bottleneck",
        "model.use_soft_regional_pooling",
        "model.use_au_region_routed",
        "model.use_global_regional_fusion",
        "model.global_regional_projection_dim",
        "model.use_adaptive_fusion_gate",
        "model.adaptive_fusion_max_alpha",
        # 2. Conservative Learning Rates
        "training.lr",
        "training.finetune_lr",
        "training.visual_extractor_lr",
        # 3. Soft Hard Semantic Loss
        "model.lambda_hard",
        "model.hard_margin",
        "model.clip_semantic.lambda_hard",
        "model.clip_semantic.hard_margin",
        "training.lambda_sem_schedule",
        # 4. Overfit monitoring & path resolution
        "training.early_stop_on_overfit",
        "training.overfit_gap_threshold",
        "training.overfit_gap_patience",
        "training.unfreeze_drop_threshold",
        "data.data_path",
        "data.sampling_strategy",
        "model.convnext_base_pretrained_path",
        # 5. Checkpoint protocol (Top-5 checkpoints requested)
        "training.max_to_keep_acc",
        "training.max_to_keep_loss",
        "training.max_to_keep_macro",
        "training.save_best_macro_f1",
        "training.mode",
    }

    unintended_diffs = []

    for key, val_b, val_c in diffs:
        print(f"{key:<42} | {str(val_b):<22} | {str(val_c):<22}")
        if not any(key.startswith(p) for p in intentional_prefixes):
            unintended_diffs.append((key, val_b, val_c))

    print("=" * 90)

    if unintended_diffs:
        print("\n[ERROR] Unintended differences detected:")
        for k, vb, vc in unintended_diffs:
            print(f"  - {k}: baseline={vb} vs candidate={vc}")
        raise AssertionError(f"Found {len(unintended_diffs)} unintended changes from 91% baseline!")
    else:
        print("\n[VERIFIED OK] Zero unintended changes! All diffs are strictly within permitted set:")
        print("  1. Direct Global + Regional Fusion added (1024-d via 4x256 projections)")
        print("  2. Conservative LR: head 1e-4, backbone 5e-6")
        print("  3. Soft Hard Semantic Loss: lambda_hard=0.02, hard_margin=0.10, lambda_sem=0.10 fixed")
        print("  4. Overfit Early Stop & Logging added")
        print("  5. Isolated Output directory: outputs/rafdb_direct_fusion_clean_91")
        print("  6. Baseline 91% augmentations, regularization (dropout=0.35, LS=0.10, WD=0.035), and seed=42 preserved 100%!")


def verify_candidate_contract(cand_path: Path):
    cfg = load_config(cand_path)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    data_cfg = cfg["data"]

    # Assertions
    assert model_cfg["use_global_regional_fusion"] is True
    assert model_cfg["global_regional_projection_dim"] == 256
    assert model_cfg["use_dynamic_part_attention"] is True
    assert model_cfg["dynamic_part_bottleneck"] == 128
    assert model_cfg["use_au_region_routed"] is True
    assert model_cfg["classifier_dropout1"] == 0.35, "Must keep baseline dropout 0.35"
    assert training_cfg["label_smoothing"] == 0.10, "Must keep baseline label smoothing 0.10"
    assert training_cfg["weight_decay"] == 0.035, "Must keep baseline weight decay 0.035"
    assert training_cfg["lr"] == 0.0001, "Head LR peak must be 1e-4"
    assert training_cfg["finetune_lr"] == 0.0001
    assert training_cfg["visual_extractor_lr"] == 0.000005, "Backbone LR must be 5e-6"
    assert model_cfg["lambda_hard"] == 0.02, "lambda_hard must be 0.02"
    assert model_cfg["hard_margin"] == 0.10, "hard_margin must be 0.10"
    assert model_cfg["lambda_sem"] == 0.10, "lambda_sem must be 0.10"
    assert aug_cfg["rotation_degrees"] == 15.0, "Must keep baseline rotation 15.0"
    assert aug_cfg["brightness_delta"] == 0.15, "Must keep baseline brightness 0.15"
    assert aug_cfg["random_erasing_prob"] == 0.40, "Must keep baseline erasing prob 0.40"
    assert data_cfg.get("sampling_strategy") in (None, "", "null"), "Must NOT use oversampling"
    assert training_cfg.get("early_stop_on_overfit") is True
    assert training_cfg.get("overfit_gap_threshold") == 0.12
    assert training_cfg.get("overfit_gap_patience") == 3

    print("\n[OK] All Candidate Contract Assertions Passed!")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default=str(ROOT / "config_rafdb_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"))
    parser.add_argument("--candidate", type=str, default=str(ROOT / "config_rafdb_direct_fusion_clean_91.yaml"))
    args = parser.parse_args()

    base_path = Path(args.baseline)
    cand_path = Path(args.candidate)

    print_and_verify_diff(base_path, cand_path)
    verify_candidate_contract(cand_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
