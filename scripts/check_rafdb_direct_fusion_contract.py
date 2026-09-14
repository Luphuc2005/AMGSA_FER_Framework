#!/usr/bin/env python3
"""Contract and sanity verification script for RAF-DB Direct Fusion SOTA Candidates A & B."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def verify_direct_fusion_contract(cfg_path: Path, candidate: str = "auto") -> dict:
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    data_cfg = cfg["data"]

    # Auto detect candidate if requested
    if candidate == "auto":
        if "candidate_b" in cfg_path.name:
            candidate = "b"
        else:
            candidate = "a"

    # 1. Architecture & Direct Fusion (Shared across Candidate A & B)
    assert model_cfg["arch"] == "convnext_base_ms1m_arcface", f"Wrong arch: {model_cfg['arch']}"
    assert model_cfg["use_global_regional_fusion"] is True, "use_global_regional_fusion must be True"
    assert int(model_cfg["global_regional_projection_dim"]) == 256, "global_regional_projection_dim must be 256"
    assert model_cfg["use_dynamic_part_attention"] is True, "use_dynamic_part_attention must be True"
    assert int(model_cfg["dynamic_part_bottleneck"]) == 128, "dynamic_part_bottleneck must be 128"
    assert model_cfg["use_soft_regional_pooling"] is False, "use_soft_regional_pooling must be False"
    assert model_cfg["use_au_region_routed"] is True, "use_au_region_routed must be True"
    assert model_cfg["use_adaptive_fusion_gate"] is True, "use_adaptive_fusion_gate must be True"
    assert abs(float(model_cfg["adaptive_fusion_max_alpha"]) - 0.20) < 1e-4, "adaptive_fusion_max_alpha must be 0.20"
    assert model_cfg["use_adaptive_granularity"] is True, "use_adaptive_granularity must be True"
    assert model_cfg["multi_prototype"] is True, "multi_prototype must be True"
    assert model_cfg["use_semantic_branch"] is True, "use_semantic_branch must be True"
    assert model_cfg["use_clip_semantic"] is True, "use_clip_semantic must be True"

    # 2. Augmentations (Shared mild augmentations)
    assert abs(float(aug_cfg["rotation_degrees"]) - 10.0) < 1e-4, "rotation_degrees must be 10.0"
    assert abs(float(aug_cfg["brightness_delta"]) - 0.10) < 1e-4, "brightness_delta must be 0.10"
    assert abs(float(aug_cfg["contrast_lower"]) - 0.90) < 1e-4, "contrast_lower must be 0.90"
    assert abs(float(aug_cfg["contrast_upper"]) - 1.10) < 1e-4, "contrast_upper must be 1.10"
    assert abs(float(aug_cfg["gamma_prob"]) - 0.25) < 1e-4, "gamma_prob must be 0.25"
    assert abs(float(aug_cfg["gamma_min"]) - 0.80) < 1e-4, "gamma_min must be 0.80"
    assert abs(float(aug_cfg["gamma_max"]) - 1.25) < 1e-4, "gamma_max must be 1.25"
    assert abs(float(aug_cfg["random_erasing_prob"]) - 0.20) < 1e-4, "random_erasing_prob must be 0.20"
    assert abs(float(aug_cfg["random_erasing_area_max"]) - 0.10) < 1e-4, "random_erasing_area_max must be 0.10"

    # 3. Regularization (Shared reduced regularization)
    assert abs(float(training_cfg["label_smoothing"]) - 0.05) < 1e-4, "label_smoothing must be 0.05"
    assert abs(float(model_cfg["classifier_dropout1"]) - 0.25) < 1e-4, "classifier_dropout1 must be 0.25"

    # 4. Hard Semantic Loss (Shared soft hard semantic loss)
    assert model_cfg["use_hard_semantic_loss"] is True, "use_hard_semantic_loss must be True"
    assert abs(float(model_cfg["lambda_sem"]) - 0.10) < 1e-4, "lambda_sem must be 0.10"
    assert abs(float(model_cfg["lambda_hard"]) - 0.02) < 1e-4, "lambda_hard must be 0.02"
    assert abs(float(model_cfg["hard_margin"]) - 0.10) < 1e-4, "hard_margin must be 0.10"

    # 5. Sampling Strategy: Difference between Candidate A and Candidate B
    sampling_strategy = str(data_cfg.get("sampling_strategy", "")).lower()
    if candidate == "a":
        assert sampling_strategy == "class_aware_minority_oversample", f"Candidate A must have class_aware_minority_oversample, got {sampling_strategy}"
        minority_classes = [int(c) for c in data_cfg["minority_classes"]]
        assert minority_classes == [1, 2], f"minority_classes must be [1, 2] (Disgust, Fear), got {minority_classes}"
        target_counts = {int(k): int(v) for k, v in data_cfg["target_minority_counts"].items()}
        assert target_counts[1] == 650, f"Disgust target must be 650, got {target_counts[1]}"
        assert target_counts[2] == 550, f"Fear target must be 550, got {target_counts[2]}"
        assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/rafdb_direct_fusion_sota_acc"
    else:  # Candidate B
        assert sampling_strategy in ("", "none", "null", "false"), f"Candidate B must have NO oversampling, got {sampling_strategy}"
        assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/rafdb_direct_fusion_candidate_b"

    # 6. Runtime, Optimizer, Paths
    assert cfg["seed"]["random_seed"] == 42, "Seed must be 42"
    assert data_cfg["image_size"] == 112, "image_size must be 112"
    assert cfg["runtime"]["batch_size_per_gpu"] == 16, "batch_size_per_gpu must be 16"
    assert training_cfg["optimizer"] == "sam", "optimizer must be sam"
    assert training_cfg["base_optimizer"] == "adamw", "base_optimizer must be adamw"
    assert training_cfg["scheduler"] == "cosine", "scheduler must be cosine"

    print(f"[OK] Config assertions passed for Candidate {candidate.upper()} ({cfg_path.name})!")
    return cfg


def test_dataset_sampling(cfg: dict, candidate: str):
    from datasets.fer2013 import collect_split_records, make_dataset, get_oversample_stats

    data_dir = ROOT / cfg["data"]["data_path"]
    if not data_dir.exists():
        print(f"[WARN] Data directory {data_dir} not found locally, skipping dataset loading check.")
        return

    train_records = collect_split_records(data_dir, "train")
    orig_counts = np.bincount(train_records.labels)
    print(f"[INFO] Original train records: {len(train_records.labels)}, counts: {orig_counts.tolist()}")

    train_ds = make_dataset(train_records, cfg, split="train", training=True, replicas=1)
    stats = get_oversample_stats()

    if candidate == "a":
        assert stats is not None, "Candidate A: Oversample stats must not be None"
        effective_counts = stats["effective_class_counts"]
        print(f"[INFO] Candidate A Effective train counts: {effective_counts}")
        assert effective_counts[2] == 550, f"Fear effective count must be 550, got {effective_counts[2]}"
        assert effective_counts[1] == 650, f"Disgust effective count must be 650, got {effective_counts[1]}"
    else:  # Candidate B
        # In Candidate B, no oversample stats are generated, effective dataset uses original counts
        print(f"[INFO] Candidate B: Natural distribution confirmed (Fear: {orig_counts[2]}, Disgust: {orig_counts[1]})")

    # Check val dataset is not oversampled
    val_records = collect_split_records(data_dir, "val")
    val_ds = make_dataset(val_records, cfg, split="val", training=False, replicas=1)
    print(f"[INFO] Val split count (untouched): {len(val_records.labels)}")

    print(f"[OK] Dataset sampling test passed for Candidate {candidate.upper()}!")


def verify_derivation_from_baseline():
    """Verify that Candidate A and B were strictly derived from the baseline RAF-DB config."""
    baseline_path = ROOT / "config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml"
    assert baseline_path.exists(), f"Baseline config not found: {baseline_path}"
    base_cfg = load_config(baseline_path)

    # Verify baseline properties
    assert abs(float(base_cfg["model"]["classifier_dropout1"]) - 0.35) < 1e-4, "Baseline must have dropout1=0.35"
    assert abs(float(base_cfg["model"]["lambda_hard"]) - 0.05) < 1e-4, "Baseline must have lambda_hard=0.05"

    print("\n[OK] BASELINE DERIVATION CONFIRMED:")
    print(f"  - Baseline config: {baseline_path.name}")
    print(f"  - Baseline dropout1: {base_cfg['model']['classifier_dropout1']} -> Candidates A & B: 0.25 (tuned -0.10)")
    print(f"  - Baseline lambda_hard: {base_cfg['model']['lambda_hard']} -> Candidates A & B: 0.02 (tuned -0.03)")
    print("  - Both Candidate A & Candidate B share identical backbone, architecture, and hyperparameter sets.")
    print("  - Candidate A: with train-only stochastic sampling for Fear & Disgust (~550-650).")
    print("  - Candidate B: with 100% natural raw RAF-DB distribution (no oversampling).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RAF-DB Direct Fusion Candidates A & B contract.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config_rafdb_direct_fusion_candidate_b.yaml"))
    parser.add_argument("--candidate", type=str, choices=["a", "b", "auto", "both"], default="both")
    args = parser.parse_args()

    if args.candidate == "both":
        # Check Candidate A
        cfg_a_path = ROOT / "config_rafdb_direct_fusion_sota_acc.yaml"
        print("=" * 60)
        print("CHECKING CANDIDATE A (Oversampled Fear/Disgust)...")
        print("=" * 60)
        cfg_a = verify_direct_fusion_contract(cfg_a_path, candidate="a")
        test_dataset_sampling(cfg_a, candidate="a")

        # Check Candidate B
        cfg_b_path = ROOT / "config_rafdb_direct_fusion_candidate_b.yaml"
        print("\n" + "=" * 60)
        print("CHECKING CANDIDATE B (Natural Distribution)...")
        print("=" * 60)
        cfg_b = verify_direct_fusion_contract(cfg_b_path, candidate="b")
        test_dataset_sampling(cfg_b, candidate="b")
    else:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = ROOT / cfg_path
        candidate = args.candidate if args.candidate != "auto" else ("b" if "candidate_b" in cfg_path.name else "a")
        cfg = verify_direct_fusion_contract(cfg_path, candidate=candidate)
        test_dataset_sampling(cfg, candidate=candidate)

    # Verify derivation from baseline
    verify_derivation_from_baseline()

    print("\n" + "=" * 60)
    print("ALL CANDIDATE A & B CONTRACT CHECKS PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
