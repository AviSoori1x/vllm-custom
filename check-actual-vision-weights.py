#!/usr/bin/env python3
"""
Check the actual vision encoder weight names in the checkpoint
"""

import torch
import safetensors.torch

# Load checkpoint
checkpoint_path = "/mnt/vast/home/avi/omni/voxtral_eyes/consolidated.safetensors"
print("Loading checkpoint...")
state_dict = safetensors.torch.load_file(checkpoint_path)

# Get vision encoder weights
vision_weights = sorted([k for k in state_dict.keys() if k.startswith("vision_encoder")])

print(f"\nFound {len(vision_weights)} vision encoder weights")
print("\nAll vision encoder weight names:")
for name in vision_weights[:30]:  # Show first 30
    print(f"  {name}")

if len(vision_weights) > 30:
    print(f"  ... and {len(vision_weights) - 30} more")

# Check if there are any transformer.layers weights
transformer_weights = [w for w in vision_weights if "transformer.layers" in w]
print(f"\n\nFound {len(transformer_weights)} weights with 'transformer.layers'")
if transformer_weights:
    print("First few:")
    for w in transformer_weights[:5]:
        print(f"  {w}")

# Check for attention weights pattern
attention_weights = [w for w in vision_weights if "attention" in w]
print(f"\n\nFound {len(attention_weights)} weights with 'attention'")
if attention_weights:
    print("First few:")
    for w in attention_weights[:5]:
        print(f"  {w}")

# Check structure more carefully
print("\n\nAnalyzing weight name structure...")
# Get unique second-level components
second_level = set()
for name in vision_weights:
    parts = name.split(".")
    if len(parts) > 2:
        second_level.add(parts[1] + "." + parts[2])

print("Second-level components:")
for comp in sorted(second_level):
    print(f"  {comp}")
