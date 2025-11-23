#!/usr/bin/env python3
"""
Inspect the prune_config stored in viz_data.pt files to understand the pruning algorithm.

Usage:
    python inspect_prune_config.py --model csp_yolo2 --dataset bracket_counting_beeg
    python inspect_prune_config.py --url <full_path_to_viz_data.pt>
"""

import argparse
import io
import json
from typing import Dict, Any

import torch
from tiktoken.load import read_file_cached

from circuit_sparsity.registries import MODEL_BASE_DIR


def inspect_viz_data(path: str) -> None:
    """Load and inspect a viz_data.pt file."""
    print(f"\n{'='*70}")
    print(f"Inspecting: {path}")
    print(f"{'='*70}\n")

    try:
        buf = io.BytesIO(read_file_cached(path))
        viz_data = torch.load(buf, map_location="cpu", weights_only=True)

        # Print top-level keys
        print("Top-level keys in viz_data:")
        for key in viz_data.keys():
            value = viz_data[key]
            if isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value).__name__} with {len(value)} elements")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__}")

        # Inspect prune_config from multiple possible locations
        prune_config = None

        # Try viz_data["prune_config"]
        if "prune_config" in viz_data:
            prune_config = viz_data["prune_config"]
            print("\n" + "="*70)
            print("PRUNE_CONFIG (from viz_data['prune_config']):")
            print("="*70)
            print(json.dumps(prune_config, indent=2, default=str))

        # Try viz_data["circuit_data"]["prune_config"]
        if "circuit_data" in viz_data and isinstance(viz_data["circuit_data"], dict):
            if "prune_config" in viz_data["circuit_data"]:
                prune_config_alt = viz_data["circuit_data"]["prune_config"]
                print("\n" + "="*70)
                print("PRUNE_CONFIG (from viz_data['circuit_data']['prune_config']):")
                print("="*70)
                print(json.dumps(prune_config_alt, indent=2, default=str))

        # Inspect importances structure
        if "importances" in viz_data:
            print("\n" + "="*70)
            print("IMPORTANCES KEYS:")
            print("="*70)
            importances = viz_data["importances"]
            for key in importances.keys():
                value = importances[key]
                if isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} keys")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} with {len(value)} elements")
                elif isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor {value.shape}")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")

        # Check for all_loss (pruning iterations)
        if "all_loss" in viz_data and viz_data["all_loss"] is not None:
            print("\n" + "="*70)
            print("PRUNING ITERATIONS (all_loss):")
            print("="*70)
            all_loss = viz_data["all_loss"]
            print(f"  Number of pruning steps: {len(all_loss)}")
            if len(all_loss) > 0:
                print(f"  Format: (step, loss)")
                for i, (step, loss) in enumerate(all_loss[:5]):
                    print(f"    Step {i}: ({step}, {loss:.6f})")
                if len(all_loss) > 5:
                    print(f"    ... ({len(all_loss) - 5} more)")

        # Inspect circuit_data structure
        if "circuit_data" in viz_data:
            print("\n" + "="*70)
            print("CIRCUIT_DATA (node masks):")
            print("="*70)
            circuit_data = viz_data["circuit_data"]
            total_nodes = 0
            for key, mask in circuit_data.items():
                if key == "prune_config":
                    continue
                if isinstance(mask, torch.Tensor):
                    n_kept = mask.sum().item() if mask.dtype == torch.bool else mask.numel()
                    total_nodes += n_kept
                    print(f"  {key}: {mask.shape} -> {n_kept} nodes kept")
            print(f"\n  Total nodes in circuit: {total_nodes}")

        # Summary
        print("\n" + "="*70)
        print("SUMMARY:")
        print("="*70)
        print(f"  Model config: {viz_data.get('importances', {}).get('beeg_model_config', 'Not found')}")
        print(f"  Baseline loss: {viz_data.get('importances', {}).get('loss', 'Not found')}")
        print(f"  Pruned loss: {viz_data.get('importances', {}).get('interv_loss', 'Not found')}")
        print(f"  Total nodes: {viz_data.get('num_total_nodes', 'Not found')}")

        if prune_config is None:
            print("\n⚠️  WARNING: No prune_config found in this file!")

    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Model name (e.g., csp_yolo2)")
    parser.add_argument("--dataset", help="Dataset name (e.g., bracket_counting_beeg)")
    parser.add_argument("--sweep", default="prune_v4", help="Sweep name (default: prune_v4)")
    parser.add_argument("--k", default="k_optim", help="k value (default: k_optim)")
    parser.add_argument("--url", help="Direct URL to viz_data.pt file")

    args = parser.parse_args()

    if args.url:
        path = args.url
    elif args.model and args.dataset:
        path = f"{MODEL_BASE_DIR}/viz/{args.model}/{args.dataset}/{args.sweep}/{args.k}/viz_data.pt"
    else:
        parser.error("Must provide either --url OR (--model AND --dataset)")

    inspect_viz_data(path)


if __name__ == "__main__":
    main()
