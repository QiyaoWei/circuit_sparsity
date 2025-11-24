#!/usr/bin/env python3
"""
Deep dive into the loss values structure to understand the relationship.
"""

import io
import torch
from tiktoken.load import read_file_cached
from circuit_sparsity.registries import MODEL_BASE_DIR

print("="*80)
print("DEEP DIVE INTO LOSS VALUE RELATIONSHIPS")
print("="*80)

# Load viz_data for k=64
viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo1/single_double_quote/prune_v2/64/viz_data.pt"
buf = io.BytesIO(read_file_cached(viz_path))
viz_data = torch.load(buf, map_location="cpu", weights_only=True)

importances = viz_data['importances']

print("\n1. MAIN LOSS VALUES")
print("="*80)
print(f"loss (baseline - full model):  {importances['loss']:.10f}")
print(f"interv_loss (circuit?):        {importances['interv_loss']:.10f}")
print(f"Ratio (interv/baseline):       {importances['interv_loss']/importances['loss']:.4f}x")

print("\n2. LAYER INTERVENTION LOSSES")
print("="*80)
if 'layer_interv_losses' in importances:
    layer_losses = importances['layer_interv_losses']
    print(f"Number of layers: {len(layer_losses)}")
    print("\nLayer losses (first 5):")
    for i, (layer_name, loss_val) in enumerate(list(layer_losses.items())[:5]):
        # Compute delta as done in viz.py line 191
        delta = loss_val - importances['interv_loss'] / 2
        print(f"  {layer_name}:")
        print(f"    Raw loss:      {loss_val:.10f}")
        print(f"    Delta:         {delta:.10f}")
        print(f"    interv_loss/2: {importances['interv_loss']/2:.10f}")
else:
    print("  No layer_interv_losses found")

print("\n3. CHANNEL INTERVENTION LOSSES")
print("="*80)
if 'ch_interv_losses' in importances:
    ch_losses = importances['ch_interv_losses']
    print(f"Number of channel locations: {len(ch_losses)}")
    print("\nChannel losses (first 3 locations):")
    for i, (loc_name, ch_list) in enumerate(list(ch_losses.items())[:3]):
        if isinstance(ch_list, list) and len(ch_list) > 0:
            # The values in the list after dividing by 2 (see viz.py line 196)
            ch_list_divided = [y / 2.0 for y in ch_list]
            print(f"  {loc_name}:")
            print(f"    Number of channels: {len(ch_list)}")
            print(f"    First 3 values (raw): {ch_list[:3]}")
            print(f"    First 3 values (÷2):  {ch_list_divided[:3]}")
else:
    print("  No ch_interv_losses found")

print("\n4. INTERPRETATION")
print("="*80)
print("""
From viz.py line 682, we know layer_imps is called "ablation delta".

The formula is:
  layer_imps[layer] = layer_interv_losses[layer] - interv_loss / 2

If layer_interv_losses are intervention costs (how much worse when you ablate):
  - Higher layer_interv_losses = ablating this layer hurts more = layer is important
  - The delta compares to interv_loss as baseline

But what IS interv_loss?

Two possibilities:
  A) interv_loss = loss with full circuit (keep circuit nodes, ablate others)
     → Then layer_interv_losses = loss when you ablate that layer FROM the circuit
     → Delta = how much that layer contributes to circuit

  B) interv_loss = loss with full model (no intervention)
     → Then layer_interv_losses = loss when you ablate that layer FROM full model
     → Delta = how important that layer is in full model

The fact that interv_loss can be LOWER than baseline (full model) suggests A is wrong...
Unless the labels are swapped?

Let me check the /2 factor...
""")

print("\n5. CHECKING THE FACTOR OF 2")
print("="*80)
print(f"interv_loss:           {importances['interv_loss']:.10f}")
print(f"interv_loss / 2:       {importances['interv_loss']/2:.10f}")
print(f"interv_loss * 2:       {importances['interv_loss']*2:.10f}")
print(f"baseline (loss):       {importances['loss']:.10f}")
print(f"baseline * 2:          {importances['loss']*2:.10f}")
print()
print(f"Does interv_loss * 2 ≈ baseline? {abs(importances['interv_loss']*2 - importances['loss']) < 0.001}")
