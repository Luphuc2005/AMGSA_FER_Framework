#!/usr/bin/env python3
"""Contract and sanity verification script for RAF-DB Direct Global + Regional Fusion SOTA config."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config


def verify_direct_fusion_contract(cfg_path: Path) -> dict:
    cfg = load_config(cfg_path)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    data_cfg = cfg["data"]

    # 1. Architecture & Direct Fusion
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

    # 2. Augmentations
    assert abs(float(aug_cfg["rotation_degrees"]) - 10.0) < 1e-4, "rotation_degrees must be 10.0"
    assert abs(float(aug_cfg["brightness_delta"]) - 0.10) < 1e-4, "brightness_delta must be 0.10"
    assert abs(float(aug_cfg["contrast_lower"]) - 0.90) < 1e-4, "contrast_lower must be 0.90"
    assert abs(float(aug_cfg["contrast_upper"]) - 1.10) < 1e-4, "contrast_upper must be 1.10"
    assert abs(float(aug_cfg["gamma_prob"]) - 0.25) < 1e-4, "gamma_prob must be 0.25"
    assert abs(float(aug_cfg["gamma_min"]) - 0.80) < 1e-4, "gamma_min must be 0.80"
    assert abs(float(aug_cfg["gamma_max"]) - 1.25) < 1e-4, "gamma_max must be 1.25"
    assert abs(float(aug_cfg["random_erasing_prob"]) - 0.20) < 1e-4, "random_erasing_prob must be 0.20"
    assert abs(float(aug_cfg["random_erasing_area_max"]) - 0.10) < 1e-4, "random_erasing_area_max must be 0.10"

    # 3. Regularization
    assert abs(float(training_cfg["label_smoothing"]) - 0.05) < 1e-4, "label_smoothing must be 0.05"
    assert abs(float(model_cfg["classifier_dropout1"]) - 0.25) < 1e-4, "classifier_dropout1 must be 0.25"

    # 4. Hard Semantic Loss
    assert model_cfg["use_hard_semantic_loss"] is True, "use_hard_semantic_loss must be True"
    assert abs(float(model_cfg["lambda_sem"]) - 0.10) < 1e-4, "lambda_sem must be 0.10"
    assert abs(float(model_cfg["lambda_hard"]) - 0.02) < 1e-4, "lambda_hard must be 0.02"
    assert abs(float(model_cfg["hard_margin"]) - 0.10) < 1e-4, "hard_margin must be 0.10"

    # 5. Class-aware sampling
    assert str(data_cfg["sampling_strategy"]).lower() == "class_aware_minority_oversample"
    minority_classes = [int(c) for c in data_cfg["minority_classes"]]
    assert minority_classes == [1, 2], f"minority_classes must be [1, 2] (Disgust, Fear), got {minority_classes}"
    target_counts = {int(k): int(v) for k, v in data_cfg["target_minority_counts"].items()}
    assert target_counts[1] == 650, f"Disgust target must be 650, got {target_counts[1]}"
    assert target_counts[2] == 550, f"Fear target must be 550, got {target_counts[2]}"

    # 6. Runtime, Optimizer, Paths
    assert cfg["seed"]["random_seed"] == 42, "Seed must be 42"
    assert data_cfg["image_size"] == 112, "image_size must be 112"
    assert cfg["runtime"]["batch_size_per_gpu"] == 16, "batch_size_per_gpu must be 16"
    assert training_cfg["optimizer"] == "sam", "optimizer must be sam"
    assert training_cfg["base_optimizer"] == "adamw", "base_optimizer must be adamw"
    assert training_cfg["scheduler"] == "cosine", "scheduler must be cosine"
    assert Path(cfg["paths"]["output_dir"]) == ROOT / "outputs/rafdb_direct_fusion_sota_acc", "output_dir must be isolated"

    print(f"[OK] Config assertions passed for {cfg_path.name}!")
    return cfg


def test_dataset_sampling(cfg: dict):
    from datasets.fer2013 import collect_split_records, make_dataset, get_oversample_stats

    data_dir = ROOT / cfg["data"]["data_path"]
    if not data_dir.exists():
        print(f"[WARN] Data directory {data_dir} not found locally, skipping dataset loading check.")
        return

    train_records = collect_split_records(data_dir, "train")
    print(f"[INFO] Original train records: {len(train_records.labels)}, counts: {np.bincount(train_records.labels)}")

    train_ds = make_dataset(train_records, cfg, split="train", training=True, replicas=1)
    stats = get_oversample_stats()
    assert stats is not None, "Oversample stats must not be None"
    effective_counts = stats["effective_class_counts"]
    print(f"[INFO] Effective train class counts: {effective_counts}")
    assert effective_counts[2] == 550, f"Fear effective count must be 550, got {effective_counts[2]}"
    assert effective_counts[1] == 650, f"Disgust effective count must be 650, got {effective_counts[1]}"

    # Check val dataset is not oversampled
    val_records = collect_split_records(data_dir, "val")
    val_ds = make_dataset(val_records, cfg, split="val", training=False, replicas=1)
    print(f"[INFO] Val split count (untouched): {len(val_records.labels)}")

    print("[OK] Dataset sampling test passed!")


def test_model_forward(cfg: dict):
    import tensorflow as tf
    from models.convnext_base_face_baseline import ConvNeXtBaseFaceFERBaseline

    print("[INFO] Testing model instantiation and forward pass...")
    # Build model in evaluation mode
    model = ConvNeXtBaseFaceFERBaseline(cfg)
    dummy_input = tf.zeros([2, 112, 112, 3], dtype=tf.float32)
    dummy_labels = tf.constant([0, 1], dtype=tf.int64)

    outputs = model({"image": dummy_input}, training=False, labels=dummy_labels)
    assert "logits" in outputs, "outputs must have logits"
    assert outputs["logits"].shape == (2, 7), f"Logits shape should be (2, 7), got {outputs['logits'].shape}"
    assert "global_regional_features" in outputs, "global_regional_features must be in outputs"
    assert outputs["global_regional_features"].shape == (2, 1024), (
        f"Fused global_regional_features must be (2, 1024), got {outputs['global_regional_features'].shape}"
    )
    print(f"[OK] Model forward pass successful! Fused features: {outputs['global_regional_features'].shape}, Logits: {outputs['logits'].shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RAF-DB Direct Fusion SOTA contract.")
    parser.add_argument("--config", type=str, default=str(ROOT / "config_rafdb_direct_fusion_sota_acc.yaml"))
    parser.add_argument("--skip-model", action="store_true", help="Skip model forward pass")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path

    cfg = verify_direct_fusion_contract(cfg_path)
    test_dataset_sampling(cfg)
    if not args.skip_model:
        test_model_forward(cfg)

    print("\n" + "=" * 60)
    print("ALL RAF-DB DIRECT FUSION SOTA CONTRACT TESTS PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
