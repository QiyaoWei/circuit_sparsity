#!/usr/bin/env python3
"""
List all available viz_data.pt files in the OpenAI circuit-sparsity dataset.
This helps understand what data is available for analysis.

Usage:
    python list_available_data.py [--model MODEL] [--verbose]
"""

import argparse
from collections import defaultdict
from typing import Dict, List

import blobfile as bf

MODEL_BASE_DIR = "https://openaipublic.blob.core.windows.net/circuit-sparsity"


def list_viz_files(model_filter: str = None, verbose: bool = False) -> Dict:
    """List all viz_data.pt files organized by model/dataset/sweep/k."""
    base_path = f"{MODEL_BASE_DIR}/viz"

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    total_count = 0

    try:
        models = list(bf.listdir(base_path))
        if model_filter:
            models = [m for m in models if model_filter in m]

        for model in sorted(models):
            if verbose:
                print(f"\n📦 Model: {model}")

            try:
                datasets = list(bf.listdir(f"{base_path}/{model}"))
                for dataset in sorted(datasets):
                    if verbose:
                        print(f"  📊 Dataset: {dataset}")

                    try:
                        sweeps = list(bf.listdir(f"{base_path}/{model}/{dataset}"))
                        for sweep in sorted(sweeps):
                            if verbose:
                                print(f"    🔬 Sweep: {sweep}")

                            try:
                                ks = list(bf.listdir(f"{base_path}/{model}/{dataset}/{sweep}"))
                                for k in sorted(ks):
                                    viz_path = f"{base_path}/{model}/{dataset}/{sweep}/{k}/viz_data.pt"
                                    if bf.exists(viz_path):
                                        results[model][dataset][sweep].append(k)
                                        total_count += 1
                                        if verbose:
                                            print(f"      ✓ k={k}")
                            except Exception as e:
                                if verbose:
                                    print(f"      Error listing k values: {e}")
                    except Exception as e:
                        if verbose:
                            print(f"    Error listing sweeps: {e}")
            except Exception as e:
                if verbose:
                    print(f"  Error listing datasets: {e}")

    except Exception as e:
        print(f"Error accessing remote data: {e}")
        return None

    return dict(results), total_count


def print_summary(results: Dict, total_count: int):
    """Print a summary of available data."""
    print("\n" + "="*70)
    print("SUMMARY OF AVAILABLE VIZ_DATA.PT FILES")
    print("="*70)

    for model, datasets in sorted(results.items()):
        print(f"\n🔹 Model: {model}")
        for dataset, sweeps in sorted(datasets.items()):
            print(f"  ├─ Dataset: {dataset}")
            for sweep, ks in sorted(sweeps.items()):
                k_str = ", ".join(str(k) for k in sorted(ks, key=lambda x: (isinstance(x, str), x)))
                print(f"  │  ├─ Sweep: {sweep}")
                print(f"  │  │  └─ k values ({len(ks)}): {k_str}")

    print(f"\n{'='*70}")
    print(f"TOTAL: {total_count} viz_data.pt files found")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Filter by model name (substring match)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("🔍 Scanning remote dataset...")
    results, total_count = list_viz_files(model_filter=args.model, verbose=args.verbose)

    if results:
        print_summary(results, total_count)
    else:
        print("❌ Could not access remote data. Check network connection.")


if __name__ == "__main__":
    main()
