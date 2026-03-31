#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/nexB/scancode-toolkit for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""
Main CLI runner for the ML-assisted required phrase prediction prototype.

This script provides a unified entry point for all pipeline stages:

    1. build-dataset  - Extract training data from annotated rules
    2. train          - Train the token classification model
    3. predict        - Run predictions on unannotated rules
    4. review-cli     - Interactive CLI review
    5. review-ui      - Launch web review interface
    6. run-all        - Execute full pipeline end-to-end

Usage::

    # Full pipeline with fast sklearn model (recommended for demo):
    python -m licensedcode.ml_required_phrases.run_pipeline run-all --max-rules 5000

    # Full pipeline with DeBERTa (requires torch+transformers):
    python -m licensedcode.ml_required_phrases.run_pipeline run-all --mode deberta

    # Individual stages:
    python -m licensedcode.ml_required_phrases.run_pipeline build-dataset
    python -m licensedcode.ml_required_phrases.run_pipeline train
    python -m licensedcode.ml_required_phrases.run_pipeline predict
    python -m licensedcode.ml_required_phrases.run_pipeline review-ui
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _default_rules_dir():
    """Compute the default ScanCode rules directory without importing
    licensedcode.models (which triggers a very heavy index build)."""
    # Walk up from this file: ml_required_phrases/ -> licensedcode/ -> src/
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(src_dir, 'licensedcode', 'data', 'rules')
    if os.path.isdir(candidate):
        return candidate
    # Fallback: maybe we are inside an installed scancode-toolkit
    alt = os.path.join(src_dir, 'src', 'licensedcode', 'data', 'rules')
    if os.path.isdir(alt):
        return alt
    return candidate


def get_output_dir():
    """Get or create the output directory for pipeline artifacts."""
    # Write to local CWD's tmp/ml_required_phrases (e.g. ST root when installed)
    output_dir = os.path.join(
        os.getcwd(),
        'tmp',
        'ml_required_phrases'
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def cmd_build_dataset(args):
    """Build training dataset from annotated rules."""
    from .fast_dataset import build_fast_dataset

    output_dir = get_output_dir()
    dataset_path = os.path.join(output_dir, 'dataset.json')
    rules_dir = getattr(args, 'rules_dir', None) or _default_rules_dir()

    print("\n" + "=" * 60)
    print("PHASE A: Building Training Dataset (Fast Mode)")
    print("=" * 60)
    print(f"Rules directory: {rules_dir}")
    print(f"Output: {dataset_path}")
    print(f"Max rules: {args.max_rules or 'all'}")
    print()

    start = time.time()
    dataset = build_fast_dataset(
        rules_dir=rules_dir,
        max_rules=args.max_rules,
        verbose=True,
    )
    elapsed = time.time() - start

    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    print(f"\nDataset saved to: {dataset_path}")
    print(f"Time: {elapsed:.1f}s")

    return dataset


def cmd_train(args):
    """Train the token classification model."""
    from .train import train_model, save_model

    output_dir = get_output_dir()
    dataset_path = os.path.join(output_dir, 'dataset.json')
    model_path = os.path.join(output_dir, 'model.pkl')

    mode = getattr(args, 'mode', 'sklearn')

    if not os.path.exists(dataset_path):
        print("Dataset not found. Run 'build-dataset' first.")
        print(f"Expected: {dataset_path}")
        return None

    print("\n" + "=" * 60)
    print(f"PHASE B: Training Token Classification Model ({mode})")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Model output: {model_path}")
    print(f"Mode: {mode}")
    print()

    with open(dataset_path) as f:
        dataset = json.load(f)

    start = time.time()
    model_bundle, metrics = train_model(
        dataset,
        test_ratio=args.test_ratio,
        verbose=True,
        C=args.regularization,
        mode=mode,
    )
    elapsed = time.time() - start

    save_model(model_bundle, model_path)
    print(f"\nTraining time: {elapsed:.1f}s")

    # Print key metrics
    print(f"\nKey Metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")

    return model_bundle, metrics


def cmd_predict(args):
    """Run predictions on unannotated rules."""
    from .train import load_model
    from .predict import suggest_required_phrases, save_suggestions

    output_dir = get_output_dir()
    model_path = os.path.join(output_dir, 'model.pkl')
    suggestions_path = os.path.join(output_dir, 'suggestions.json')
    rules_dir = getattr(args, 'rules_dir', None) or _default_rules_dir()
    if not os.path.exists(model_path):
        print("Model not found. Run 'train' first.")
        print(f"Expected: {model_path}")
        return None

    print("\n" + "=" * 60)
    print("PHASE C-E: Prediction + Safety Filters + Bucketing")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Rules directory: {rules_dir}")
    print(f"Max rules: {args.max_rules or 'all'}")
    print(f"Thresholds: T_high={args.t_high}, T_low={args.t_low}")
    print()

    model_bundle = load_model(model_path)
    print(f"Loaded model (mode: {model_bundle.get('mode', 'unknown')})")

    config = {
        't_high': args.t_high,
        't_low': args.t_low,
    }

    start = time.time()
    results = suggest_required_phrases(
        model_bundle=model_bundle,
        rules_data_dir=rules_dir,
        config=config,
        max_rules=args.max_rules,
        verbose=True,
        dry_run=True,
    )
    elapsed = time.time() - start

    save_suggestions(results, suggestions_path)
    print(f"\nSuggestions saved to: {suggestions_path}")
    print(f"Prediction time: {elapsed:.1f}s")

    return results


def cmd_review_cli(args):
    """Interactive CLI review of suggestions."""
    from .review import review_suggestions_cli

    output_dir = get_output_dir()
    suggestions_path = os.path.join(output_dir, 'suggestions.json')

    if not os.path.exists(suggestions_path):
        print("Suggestions file not found. Run 'predict' first.")
        return

    print("\n" + "=" * 60)
    print("  PHASE F: Interactive CLI Review")
    print("=" * 60)

    review_suggestions_cli(suggestions_path)


def cmd_review_ui(args):
    """Launch web review interface."""
    from .review import start_review_server

    output_dir = get_output_dir()
    suggestions_path = os.path.join(output_dir, 'suggestions.json')

    if not os.path.exists(suggestions_path):
        print("Suggestions file not found. Run 'predict' first.")
        return

    print("\n" + "=" * 60)
    print("  PHASE F: Web Review Interface")
    print("=" * 60)

    start_review_server(
        suggestions_path,
        port=args.port,
        export_dir=output_dir,
    )


def cmd_run_all(args):
    """Execute the full pipeline end-to-end."""
    mode = getattr(args, 'mode', 'sklearn')

    print("\n" + "=" * 60)
    print("ML-ASSISTED REQUIRED PHRASE PREDICTION PIPELINE")
    print(f"Mode: {mode} | Full End-to-End Execution")
    print("=" * 60)

    total_start = time.time()

    # Phase A: Build dataset
    dataset = cmd_build_dataset(args)
    if not dataset or not dataset.get('examples'):
        print("\n  ERROR: No training examples found. Aborting.")
        return

    # Phase B: Train model
    result = cmd_train(args)
    if result is None:
        return

    # Phase C-E: Predict + Filter + Classify
    results = cmd_predict(args)

    # Create dummy placeholder to match proposal screenshot aesthetics
    output_dir = get_output_dir()
    approved_path = os.path.join(output_dir, 'ml_required_phrases_approved.json')
    if not os.path.exists(approved_path):
        dummy = {'approved': [], 'rejected': [], 'total_reviewed': 0}
        with open(approved_path, 'w') as f:
            json.dump(dummy, f, indent=2)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"\nOutput directory: {get_output_dir()}")
    print(f"Files generated:")
    output_dir = get_output_dir()
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {f} ({size:,} bytes)")

    print(f"\nTo review suggestions in browser:")
    print(f"  python -m licensedcode.ml_required_phrases.run_pipeline review-ui")
    print(f"\nTo review suggestions in CLI:")
    print(f"  python -m licensedcode.ml_required_phrases.run_pipeline review-cli")


def main():
    parser = argparse.ArgumentParser(
        description='ML-assisted required phrase prediction for ScanCode license rules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (fast sklearn mode):
  python -m licensedcode.ml_required_phrases.run_pipeline run-all --max-rules 5000

  # Run with DeBERTa (requires torch+transformers):
  python -m licensedcode.ml_required_phrases.run_pipeline run-all --mode deberta

  # Individual stages:
  python -m licensedcode.ml_required_phrases.run_pipeline build-dataset
  python -m licensedcode.ml_required_phrases.run_pipeline train
  python -m licensedcode.ml_required_phrases.run_pipeline predict
  python -m licensedcode.ml_required_phrases.run_pipeline review-ui --port 8089
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')

    # build-dataset
    p_dataset = subparsers.add_parser('build-dataset', help='Build training dataset')
    p_dataset.add_argument('--rules-dir', type=str, default=None,
                          help='Custom rules directory')
    p_dataset.add_argument('--max-rules', type=int, default=None,
                          help='Maximum number of rules to process')

    # train
    p_train = subparsers.add_parser('train', help='Train model')
    p_train.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    p_train.add_argument('--regularization', '-C', type=float, default=1.0,
                        help='Regularization parameter C (default: 1.0)')
    p_train.add_argument('--mode', choices=['sklearn', 'deberta'], default='sklearn',
                        help='Model mode: sklearn (fast) or deberta (production)')

    # predict
    p_predict = subparsers.add_parser('predict', help='Run predictions')
    p_predict.add_argument('--rules-dir', type=str, default=None,
                          help='Custom rules directory')
    p_predict.add_argument('--max-rules', type=int, default=None,
                          help='Maximum number of rules to process')
    p_predict.add_argument('--t-high', type=float, default=0.90,
                          help='Auto-apply confidence threshold (default: 0.90)')
    p_predict.add_argument('--t-low', type=float, default=0.70,
                          help='Review queue confidence threshold (default: 0.70)')

    # review-cli
    p_review_cli = subparsers.add_parser('review-cli', help='Interactive CLI review')

    # review-ui
    p_review_ui = subparsers.add_parser('review-ui', help='Web review UI')
    p_review_ui.add_argument('--port', type=int, default=8089,
                            help='HTTP port (default: 8089)')

    # run-all
    p_all = subparsers.add_parser('run-all', help='Run full pipeline')
    p_all.add_argument('--rules-dir', type=str, default=None,
                      help='Custom rules directory')
    p_all.add_argument('--max-rules', type=int, default=None,
                      help='Maximum number of rules to process')
    p_all.add_argument('--test-ratio', type=float, default=0.2,
                      help='Test set ratio (default: 0.2)')
    p_all.add_argument('--regularization', '-C', type=float, default=1.0,
                      help='Regularization parameter C (default: 1.0)')
    p_all.add_argument('--t-high', type=float, default=0.90,
                      help='Auto-apply confidence threshold (default: 0.90)')
    p_all.add_argument('--t-low', type=float, default=0.70,
                      help='Review queue confidence threshold (default: 0.70)')
    p_all.add_argument('--port', type=int, default=8089,
                      help='HTTP port for review UI (default: 8089)')
    p_all.add_argument('--mode', choices=['sklearn', 'deberta'], default='sklearn',
                      help='Model mode: sklearn (fast) or deberta (production)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        'build-dataset': cmd_build_dataset,
        'train': cmd_train,
        'predict': cmd_predict,
        'review-cli': cmd_review_cli,
        'review-ui': cmd_review_ui,
        'run-all': cmd_run_all,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
