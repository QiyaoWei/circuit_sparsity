#!/usr/bin/env python3
"""
Investigate what 'interv_loss' actually means by looking at the paper and code.

From the paper (https://arxiv.org/pdf/2511.13653):
- Section 2.2: "20 simple Python next-token binary prediction tasks"
- We need to check what the loss function is during pruning vs evaluation
"""

import io
import torch
from tiktoken.load import read_file_cached
from circuit_sparsity.registries import MODEL_BASE_DIR

print("="*80)
print("INVESTIGATING INTERV_LOSS MEANING")
print("="*80)

# Load different k values
k_values = [64, 128, 256, 512, 1024]

print("\nHypothesis: Maybe interv_loss is NOT task performance")
print("Instead it might be:")
print("  1. Ablation cost (how much behavior changes)")
print("  2. KL divergence between circuit and full model")
print("  3. Regularization term")
print("  4. Different metric entirely")

print("\n" + "="*80)
print("CHECKING VIZ_DATA CONTENTS")
print("="*80)

for k in k_values[:2]:  # Just check first 2
    viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo1/single_double_quote/prune_v2/{k}/viz_data.pt"

    try:
        buf = io.BytesIO(read_file_cached(viz_path))
        viz_data = torch.load(buf, map_location="cpu", weights_only=True)

        print(f"\nk={k}:")
        print(f"  Keys in importances: {list(viz_data['importances'].keys())}")

        # Check if there are other loss-related fields
        for key in viz_data['importances'].keys():
            if 'loss' in key.lower():
                value = viz_data['importances'][key]
                if isinstance(value, (int, float, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value
                    print(f"    {key}: {value}")

    except Exception as e:
        print(f"k={k}: Error - {e}")

print("\n" + "="*80)
print("HYPOTHESIS TEST")
print("="*80)

print("""
Looking at the values:
  Baseline: 0.001426 (constant across all k)
  k=512:    0.0000000027 (1000x lower!)

If interv_loss were task performance of the pruned circuit:
  ✗ Circuit cannot outperform full model
  ✗ Makes no physical sense

Alternative interpretations:
  1. interv_loss = loss when you ABLATE (remove) the circuit nodes
     → Lower means circuit is less important (bad!)
     → But k=512 has very low loss, suggesting it's very UNimportant??

  2. interv_loss = How well circuit ALONE performs
     → But then baseline should be higher, not lower

  3. interv_loss = Different metric entirely (KL, regularization, etc.)
     → Need to check paper/code for definition

  4. interv_loss is bugged or we're misreading the data structure

Let me check the paper for the exact definition...
""")

print("\n" + "="*80)
print("FROM THE PAPER")
print("="*80)
print("""
According to the paper (Appendix A.5):
  "Linear combination of task cross entropy and k"
  - Pruning uses: task_loss + k_coef * k
  - Target loss: 0.15 unless otherwise specified

This suggests:
  - Baseline loss = task loss on full model
  - Circuit loss during training = task_loss + regularization
  - But what is stored in interv_loss?

Need to check:
  1. Is interv_loss the task loss only, or task_loss + k_coef * k?
  2. Is there division by 2 somewhere (code says "off by factor of 2")?
  3. What samples is it evaluated on?
""")

# Check if there's a factor of 2 issue
print("\n" + "="*80)
print("CHECKING FACTOR OF 2 ISSUE")
print("="*80)

viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo1/single_double_quote/prune_v2/64/viz_data.pt"
buf = io.BytesIO(read_file_cached(viz_path))
viz_data = torch.load(buf, map_location="cpu", weights_only=True)

baseline = viz_data['importances']['loss']
interv = viz_data['importances']['interv_loss']

print(f"Baseline loss: {baseline}")
print(f"Interv loss:   {interv}")
print(f"Interv * 2:    {interv * 2}")
print(f"Baseline * 2:  {baseline * 2}")
print(f"\nRatio (interv/baseline): {interv/baseline:.4f}")
print(f"Ratio (baseline/interv): {baseline/interv:.4f}")

print("\n" + "="*80)
print("NEXT STEP")
print("="*80)
print("""
We need to look at the paper more carefully to understand:
1. What metric is used for evaluation (Section 3?)
2. How faithfulness is measured (likely in Methods)
3. Whether there are different loss functions for training vs evaluation

Paper link: https://arxiv.org/pdf/2511.13653
""")
