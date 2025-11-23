#!/usr/bin/env python3
"""
Inspect task samples from viz_data.pt files.
Shows the actual input examples used for circuit discovery.

Usage:
    python inspect_task_samples.py --model csp_yolo1 --dataset single_double_quote
    python inspect_task_samples.py --model csp_yolo2 --dataset bracket_counting_beeg
"""

import argparse
import io

import torch
from tiktoken import Encoding
from tiktoken.load import read_file_cached

from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


def truncate_zeros(sample: torch.Tensor) -> torch.Tensor:
    """Remove trailing zeros (padding tokens)."""
    assert sample.ndim == 1
    non_zero_indices = sample.nonzero()
    if non_zero_indices.numel() == 0:
        return sample
    return sample[: non_zero_indices[-1] + 1]


def inspect_samples(model_name: str, dataset_name: str, sweep: str = "prune_v4", k: str = "128"):
    """Load and display task samples from a viz_data.pt file."""

    # Try different k values if the specified one doesn't exist
    k_values = [k, "k_optim", "128", "256", "64"]

    viz_data = None
    used_k = None

    for try_k in k_values:
        viz_path = f"{MODEL_BASE_DIR}/viz/{model_name}/{dataset_name}/{sweep}/{try_k}/viz_data.pt"
        print(f"Trying: {viz_path}")
        try:
            buf = io.BytesIO(read_file_cached(viz_path))
            viz_data = torch.load(buf, map_location="cpu", weights_only=True)
            used_k = try_k
            print(f"✓ Loaded successfully!\n")
            break
        except Exception as e:
            print(f"  Failed: {e}\n")
            continue

    if viz_data is None:
        print("❌ Could not load viz_data.pt")
        return

    # Get encoding
    enc = Encoding(**tinypython.tinypython_2k())

    print("="*80)
    print(f"TASK: {dataset_name} (model: {model_name}, k={used_k})")
    print("="*80)

    # Extract task samples
    task_samples = viz_data["importances"]["task_samples"]
    print(f"\nNumber of task samples: {len(task_samples)}")

    # Show info about sample structure
    first_sample = task_samples[0]
    if isinstance(first_sample, (tuple, list)):
        print(f"Sample structure: {len(first_sample)}-tuple/list")
        print(f"  Element 0 shape: {first_sample[0].shape if hasattr(first_sample[0], 'shape') else type(first_sample[0])}")
        if len(first_sample) > 1:
            print(f"  Element 1 shape: {first_sample[1].shape if hasattr(first_sample[1], 'shape') else type(first_sample[1])}")
    else:
        print(f"Sample structure: {first_sample.shape if hasattr(first_sample, 'shape') else type(first_sample)}")

    # Display samples
    print("\n" + "="*80)
    print("TASK SAMPLES (decoded):")
    print("="*80)

    num_to_show = min(10, len(task_samples))

    for i, sample in enumerate(task_samples[:num_to_show]):
        # Extract tokens (handle both tuple and tensor formats)
        if isinstance(sample, (tuple, list)):
            tokens = sample[0]
        else:
            tokens = sample

        # Clean up tokens
        tokens_clean = truncate_zeros(tokens.flatten())
        token_list = tokens_clean.tolist()

        # Decode
        try:
            text = enc.decode(token_list)
        except:
            text = f"<decode error for tokens: {token_list[:20]}...>"

        print(f"\n{'─'*80}")
        print(f"Sample {i+1}:")
        print(f"{'─'*80}")
        print(f"Length: {len(token_list)} tokens")
        print(f"Token IDs: {token_list[:30]}{'...' if len(token_list) > 30 else ''}")
        print(f"\nDecoded text:")
        print(f"{repr(text[:300])}")
        if len(text) > 300:
            print(f"... (truncated, full length: {len(text)} chars)")

    if len(task_samples) > num_to_show:
        print(f"\n... ({len(task_samples) - num_to_show} more samples not shown)")

    # Show loss information
    print("\n" + "="*80)
    print("MODEL PERFORMANCE:")
    print("="*80)
    print(f"Baseline loss (full model): {viz_data['importances'].get('loss', 'N/A')}")
    print(f"Pruned loss (circuit): {viz_data['importances'].get('interv_loss', 'N/A')}")

    circuit_data = viz_data.get("circuit_data", {})
    total_nodes = sum(v.sum().item() if isinstance(v, torch.Tensor) and v.dtype == torch.bool
                     else v.numel()
                     for k, v in circuit_data.items()
                     if k != "prune_config" and isinstance(v, torch.Tensor))
    print(f"Total nodes in circuit: {total_nodes}")
    print(f"Total nodes in model: {viz_data.get('num_total_nodes', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model name (e.g., csp_yolo1, csp_yolo2)")
    parser.add_argument("--dataset", required=True, help="Dataset/task name")
    parser.add_argument("--sweep", default="prune_v4", help="Sweep name (default: prune_v4)")
    parser.add_argument("--k", default="128", help="k value (default: 128)")

    args = parser.parse_args()

    inspect_samples(args.model, args.dataset, args.sweep, args.k)


if __name__ == "__main__":
    main()
