#!/usr/bin/env python3
"""
Test actual task accuracy for circuits at different k values.
This will tell us if circuits really outperform the baseline or if interv_loss means something else.
"""

import io
import torch
import torch.nn.functional as F
from tiktoken.load import read_file_cached
from tiktoken import Encoding

from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.inference.hook_utils import hook_recorder
from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


def truncate_zeros(sample):
    """Remove padding zeros."""
    non_zero = sample.nonzero()
    if non_zero.numel() > 0:
        return sample[:non_zero[-1] + 1]
    return sample


def apply_circuit_mask(activations, circuit_data):
    """
    Zero out activations for nodes NOT in the circuit.

    activations: dict from hook_recorder with keys like "0.mlp.post_act"
    circuit_data: dict with channel indices to KEEP

    Returns: modified activations dict
    """
    masked_activations = {}

    for key, act_tensor in activations.items():
        if key in circuit_data and len(circuit_data[key]) > 0:
            # This location has circuit nodes
            kept_indices = circuit_data[key]

            # Create mask: all zeros except for kept indices
            mask = torch.zeros_like(act_tensor)

            # Set kept channels to 1
            if act_tensor.ndim == 3:  # [batch, seq, channels]
                mask[:, :, kept_indices] = 1.0
            elif act_tensor.ndim == 2:  # [seq, channels]
                mask[:, kept_indices] = 1.0
            else:
                print(f"Warning: unexpected shape for {key}: {act_tensor.shape}")
                mask = torch.ones_like(act_tensor)

            masked_activations[key] = act_tensor * mask
        else:
            # This location has no circuit nodes - zero everything
            masked_activations[key] = torch.zeros_like(act_tensor)

    return masked_activations


print("="*80)
print("TESTING CIRCUIT ACCURACY AT DIFFERENT K VALUES")
print("="*80)

# Load tokenizer
enc = Encoding(**tinypython.tinypython_2k())

# Target tokens
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

# Load base viz_data for samples
print("\nLoading task samples...")
viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo1/single_double_quote/prune_v2/64/viz_data.pt"
buf = io.BytesIO(read_file_cached(viz_path))
base_viz_data = torch.load(buf, map_location="cpu", weights_only=True)
task_samples = base_viz_data['importances']['task_samples'][0]
print(f"✓ Loaded {len(task_samples)} samples")

# Test full model first
print("\n" + "="*80)
print("FULL MODEL ACCURACY (no pruning)")
print("="*80)

correct_full = 0
with torch.no_grad():
    for pair_idx in range(16):
        # Double quote version
        sample_double = truncate_zeros(task_samples[pair_idx]).unsqueeze(0)
        logits_double, _, _ = model(sample_double)
        pred_double = logits_double[0, -1].argmax().item()

        if pred_double == double_quote_paren:
            correct_full += 1

        # Single quote version
        sample_single = truncate_zeros(task_samples[pair_idx + 16]).unsqueeze(0)
        logits_single, _, _ = model(sample_single)
        pred_single = logits_single[0, -1].argmax().item()

        if pred_single == single_quote_paren:
            correct_full += 1

accuracy_full = correct_full / 32
print(f"Full model accuracy: {accuracy_full:.4f} ({correct_full}/32)")

# Test circuits at different k values
print("\n" + "="*80)
print("CIRCUIT ACCURACY AT DIFFERENT K VALUES")
print("="*80)

k_values = [64, 128, 256, 512, 1024]

for k in k_values:
    viz_path_k = f"{MODEL_BASE_DIR}/viz/csp_yolo1/single_double_quote/prune_v2/{k}/viz_data.pt"

    try:
        buf = io.BytesIO(read_file_cached(viz_path_k))
        viz_data_k = torch.load(buf, map_location="cpu", weights_only=True)

        circuit_data = viz_data_k['circuit_data']

        # Count circuit nodes
        total_circuit_nodes = sum(
            len(v) for k_name, v in circuit_data.items()
            if k_name != 'prune_config' and isinstance(v, torch.Tensor)
        )

        print(f"\nk={k} ({total_circuit_nodes} circuit nodes):")
        print(f"  Baseline loss: {viz_data_k['importances']['loss']:.10f}")
        print(f"  Circuit loss:  {viz_data_k['importances']['interv_loss']:.10f}")

        # NOTE: We can't actually apply circuit masks without implementing
        # activation intervention during forward pass. The model doesn't
        # natively support masking activations.
        #
        # What we CAN do is check if the stored activations exist and are correct

        # For now, just report what we know
        print(f"  ⚠ Cannot test circuit accuracy without intervention mechanism")
        print(f"     (Need to implement activation masking during forward pass)")

    except Exception as e:
        print(f"\nk={k}: Error loading - {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
To properly test circuit accuracy, we need to:
1. Implement activation intervention during forward pass
2. OR use the stored activations from viz_data['samples']
3. OR implement the pruning mechanism from scratch

The 'interv_loss' values being lower than baseline suggest they might:
- Use a different metric (not task loss)
- Be measuring something else (ablation cost, KL divergence, etc.)
- Have a bug or misunderstanding in our interpretation
""")
