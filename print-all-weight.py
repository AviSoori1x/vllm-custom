#!/usr/bin/env python3
"""
Print all weight keys to understand the checkpoint structure
"""

import torch
import safetensors.torch

# Load checkpoint
checkpoint_path = "/mnt/vast/home/avi/omni/voxtral_eyes/consolidated.safetensors"
print("Loading checkpoint...")
state_dict = safetensors.torch.load_file(checkpoint_path)

print(f"\nTotal number of weights: {len(state_dict)}")

# Group weights by prefix
prefixes = {}
for name in state_dict.keys():
    prefix = name.split('.')[0] if '.' in name else name
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(name)

print("\nWeight prefixes and counts:")
for prefix, names in sorted(prefixes.items()):
    print(f"  {prefix}: {len(names)} weights")

# Print all weight names grouped by prefix
for prefix, names in sorted(prefixes.items()):
    print(f"\n\n=== {prefix} weights ({len(names)} total) ===")
    # Show first 20 weights for each prefix
    for i, name in enumerate(sorted(names)[:20]):
        shape = state_dict[name].shape
        print(f"  {name}: {shape}")
    if len(names) > 20:
        print(f"  ... and {len(names) - 20} more")

# Look for vision-related weights
print("\n\n=== Looking for vision-related weights ===")
vision_related = []
for name in state_dict.keys():
    if any(keyword in name.lower() for keyword in ['vision', 'image', 'visual', 'patch', 'pixel']):
        vision_related.append(name)

if vision_related:
    print(f"Found {len(vision_related)} vision-related weights:")
    for name in sorted(vision_related)[:20]:
        shape = state_dict[name].shape
        print(f"  {name}: {shape}")
else:
    print("No obvious vision-related weights found")

# Look for multimodal embeddings
print("\n\n=== Looking for mm_embeddings ===")
mm_weights = [name for name in state_dict.keys() if 'mm_embeddings' in name]
if mm_weights:
    print(f"Found {len(mm_weights)} mm_embeddings weights:")
    for name in mm_weights:
        shape = state_dict[name].shape
        print(f"  {name}: {shape}")
else:
    print("No mm_embeddings weights found")
