#!/usr/bin/env python3
"""
Compute baseline task loss for single_double_quote task (CORRECTED VERSION).

Based on analysis, we discovered:
- samples[0:16] use double quotes → should predict ")
- samples[16:32] use single quotes (same code as 0:16) → should predict ')
- Two-sided loss compares both versions
"""

import io
import torch
import torch.nn.functional as F
from tiktoken.load import read_file_cached
from tiktoken import Encoding

from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


def truncate_zeros(sample):
    """Remove padding zeros."""
    non_zero = sample.nonzero()
    if non_zero.numel() > 0:
        return sample[:non_zero[-1] + 1]
    return sample


print("="*80)
print("COMPUTING BASELINE LOSS (CORRECTED - TWO-SIDED)")
print("="*80)

# Load tokenizer
enc = Encoding(**tinypython.tinypython_2k())

# Find target tokens
single_quote_paren = enc.encode("')")[0]
double_quote_paren = enc.encode('")')[0]

print(f"\nTarget tokens:")
print(f"  ')  → {single_quote_paren}")
print(f"  \")  → {double_quote_paren}")

# Load model
print("\nLoading model...")
model_path = f"{MODEL_BASE_DIR}/models/csp_yolo1"
model = load_model(model_path, flash=True, cuda=False)
print("✓ Model loaded")

# Load viz_data
print("\nLoading viz_data...")
viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo1/single_double_quote/prune_v2/64/viz_data.pt"
buf = io.BytesIO(read_file_cached(viz_path))
viz_data = torch.load(buf, map_location="cpu", weights_only=True)
stored_loss = viz_data['importances']['loss']
print(f"✓ Stored baseline loss: {stored_loss:.10f}")

# Extract task samples
task_samples = viz_data['importances']['task_samples'][0]
print(f"✓ Found {len(task_samples)} samples")

# Verify pairing structure
assert len(task_samples) == 32, "Expected 32 samples"
print("\nSample pairing structure:")
print("  samples[0:16]  → double quote versions")
print("  samples[16:32] → single quote versions (paired with 0:16)")

# Compute baseline loss using two-sided formulation
print("\n" + "="*80)
print("COMPUTING TWO-SIDED BASELINE LOSS")
print("="*80)

total_loss = 0.0
num_pairs = 16

with torch.no_grad():
    for i in range(num_pairs):
        # Get paired samples
        double_sample = truncate_zeros(task_samples[i]).unsqueeze(0)
        single_sample = truncate_zeros(task_samples[i + 16]).unsqueeze(0)

        # Run model on both
        logits_double, _, _ = model(double_sample)
        logits_single, _, _ = model(single_sample)

        # Get last token logits
        last_logits_double = logits_double[0, -1]
        last_logits_single = logits_single[0, -1]

        # Two-sided loss formulation (matching registries.py lines 56-58)
        # For this task:
        #   tokid = double_quote_paren (target for double version)
        #   alt_tokid = single_quote_paren (target for single version)

        # Compute log softmax over just the two target tokens
        logps_double = F.log_softmax(
            last_logits_double[[double_quote_paren, single_quote_paren]],
            dim=-1
        )
        logps_single = F.log_softmax(
            last_logits_single[[double_quote_paren, single_quote_paren]],
            dim=-1
        )

        # Two-sided loss: -(log P(") | double_version) + log P(') | single_version))
        # logps_double[0] = log P(")
        # logps_single[1] = log P(')
        loss = -(logps_double[0] + logps_single[1])

        total_loss += loss.item()

        if i < 3:  # Show first 3 pairs
            print(f"\nPair {i}:")
            print(f"  Double version → P(\") = {torch.exp(logps_double[0]):.6f}")
            print(f"  Single version → P(') = {torch.exp(logps_single[1]):.6f}")
            print(f"  Loss: {loss.item():.10f}")

avg_loss = total_loss / num_pairs

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Computed average loss: {avg_loss:.10f}")
print(f"Stored baseline loss:  {stored_loss:.10f}")
print(f"Difference:            {abs(avg_loss - stored_loss):.10f}")
print(f"\nMatch? {abs(avg_loss - stored_loss) < 1e-6}")
print("="*80)
