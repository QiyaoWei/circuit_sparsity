#!/usr/bin/env python3
"""
Analyze task_samples to determine if they are paired counterfactuals.
For single_double_quote task, check if samples come in pairs with different quote types.
"""

import io
import torch
from tiktoken import Encoding
from tiktoken.load import read_file_cached

from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


def truncate_zeros(sample: torch.Tensor) -> torch.Tensor:
    """Remove trailing zeros (padding)."""
    assert sample.ndim == 1
    non_zero_indices = sample.nonzero()
    if non_zero_indices.numel() == 0:
        return sample
    return sample[: non_zero_indices[-1] + 1]


def count_quote_types(text: str) -> dict:
    """Count occurrences of single and double quotes in text."""
    # Count string delimiters (not escaped quotes inside strings)
    single_quotes = 0
    double_quotes = 0

    i = 0
    while i < len(text):
        if text[i] == "'":
            single_quotes += 1
            # Skip the string content
            i += 1
            while i < len(text) and text[i] != "'":
                if text[i] == '\\':
                    i += 2  # Skip escaped character
                else:
                    i += 1
        elif text[i] == '"':
            double_quotes += 1
            # Skip the string content
            i += 1
            while i < len(text) and text[i] != '"':
                if text[i] == '\\':
                    i += 2  # Skip escaped character
                else:
                    i += 1
        i += 1

    return {'single': single_quotes, 'double': double_quotes}


def find_string_literals(text: str) -> list:
    """Extract string literals from Python code."""
    strings = []
    i = 0
    while i < len(text):
        if text[i] in ["'", '"']:
            quote = text[i]
            start = i
            i += 1
            string_content = ""
            while i < len(text) and text[i] != quote:
                if text[i] == '\\' and i + 1 < len(text):
                    string_content += text[i:i+2]
                    i += 2
                else:
                    string_content += text[i]
                    i += 1
            if i < len(text):
                strings.append({
                    'quote_type': quote,
                    'content': string_content,
                    'full': text[start:i+1]
                })
                i += 1
        else:
            i += 1
    return strings


def compare_samples(sample1_text: str, sample2_text: str) -> dict:
    """Compare two samples to see if they're counterfactual pairs."""
    # Normalize: replace all single quotes with a marker, double quotes with another
    norm1 = sample1_text.replace("'", "QUOTE").replace('"', 'QUOTE')
    norm2 = sample2_text.replace("'", "QUOTE").replace('"', 'QUOTE')

    return {
        'identical_structure': norm1 == norm2,
        'edit_distance': sum(1 for a, b in zip(sample1_text, sample2_text) if a != b),
        'length_diff': abs(len(sample1_text) - len(sample2_text))
    }


def main():
    print("="*80)
    print("ANALYZING TASK SAMPLES FOR PAIRED STRUCTURE")
    print("="*80)

    # Load tokenizer
    enc = Encoding(**tinypython.tinypython_2k())

    # Load viz_data
    viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo1/single_double_quote/prune_v2/64/viz_data.pt"
    print(f"\nLoading: {viz_path}")
    buf = io.BytesIO(read_file_cached(viz_path))
    viz_data = torch.load(buf, map_location="cpu", weights_only=True)

    task_samples = viz_data['importances']['task_samples'][0]
    print(f"Number of samples: {len(task_samples)}")

    # Decode all samples
    print("\n" + "="*80)
    print("DECODING ALL SAMPLES")
    print("="*80)

    decoded_samples = []
    for i, sample in enumerate(task_samples):
        clean = truncate_zeros(sample)
        text = enc.decode(clean.tolist())
        decoded_samples.append(text)

        # Analyze quote usage
        strings = find_string_literals(text)
        quotes = count_quote_types(text)

        print(f"\n{'─'*80}")
        print(f"Sample {i}:")
        print(f"{'─'*80}")
        print(f"Length: {len(clean)} tokens")
        print(f"Quote counts: {quotes}")
        if strings:
            print(f"String literals ({len(strings)}):")
            for s in strings[:3]:  # Show first 3
                print(f"  {s['quote_type']}{s['content']}{s['quote_type']}")
        print(f"\nLast 150 chars of code:")
        print(repr(text[-150:]))

    # Look for paired structure
    print("\n" + "="*80)
    print("TESTING PAIRED COUNTERFACTUAL HYPOTHESIS")
    print("="*80)

    # Test if samples come in pairs (even/odd indices)
    if len(decoded_samples) % 2 == 0:
        print(f"\n✓ Even number of samples ({len(decoded_samples)}) - pairing is possible\n")

        for i in range(0, min(6, len(decoded_samples)), 2):
            print(f"\nPair {i//2}: Samples {i} and {i+1}")
            print("─"*80)

            comparison = compare_samples(decoded_samples[i], decoded_samples[i+1])
            print(f"Identical structure (quotes normalized): {comparison['identical_structure']}")
            print(f"Character differences: {comparison['edit_distance']}")
            print(f"Length difference: {comparison['length_diff']}")

            if comparison['identical_structure']:
                print("✓ These samples are COUNTERFACTUAL PAIRS!")
            elif comparison['edit_distance'] < 10:
                print("⚠ These samples are very similar but not perfect pairs")
            else:
                print("✗ These samples are NOT pairs")
    else:
        print(f"\n✗ Odd number of samples ({len(decoded_samples)}) - not obviously paired")

    # Analyze quote type distribution
    print("\n" + "="*80)
    print("QUOTE TYPE DISTRIBUTION")
    print("="*80)

    single_only = 0
    double_only = 0
    mixed = 0

    for i, text in enumerate(decoded_samples):
        quotes = count_quote_types(text)
        if quotes['single'] > 0 and quotes['double'] == 0:
            single_only += 1
            category = "single only"
        elif quotes['double'] > 0 and quotes['single'] == 0:
            double_only += 1
            category = "double only"
        else:
            mixed += 1
            category = "mixed"

        if i < 10:  # Show first 10
            print(f"Sample {i}: {quotes} -> {category}")

    print(f"\nSummary:")
    print(f"  Single quotes only: {single_only}")
    print(f"  Double quotes only: {double_only}")
    print(f"  Mixed quotes: {mixed}")

    # Final hypothesis
    print("\n" + "="*80)
    print("HYPOTHESIS")
    print("="*80)

    if single_only + double_only == len(decoded_samples):
        print("✓ Each sample uses exclusively one quote type")
        print("\nLikely structure:")
        print("  - inputs: code with one quote type")
        print("  - Should predict the matching closing quote")
        print("  - No paired counterfactuals in task_samples")
        print("\nFor two-sided loss, patch_from_inputs probably:")
        print("  - Swaps quote types on the fly, OR")
        print("  - Uses a separate dataset not stored in viz_data")
    else:
        print("⚠ Samples have mixed quote types - need deeper analysis")


if __name__ == "__main__":
    main()
