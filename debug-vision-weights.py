#!/usr/bin/env python3
"""
Debug script to check vision encoder weight names and structure
"""

import torch
import safetensors.torch
from vllm.model_executor.models.pixtral import VisionTransformer, VisionEncoderArgs

# Load checkpoint
checkpoint_path = "/mnt/vast/home/avi/omni/voxtral_eyes/consolidated.safetensors"
print("Loading checkpoint...")
state_dict = safetensors.torch.load_file(checkpoint_path)

# Get vision encoder weights
vision_weights = {k: v for k, v in state_dict.items() if k.startswith("vision_encoder")}
print(f"\nFound {len(vision_weights)} vision encoder weights")

# Print first 10 vision encoder weight names
print("\nFirst 10 vision encoder weight names:")
for i, name in enumerate(sorted(vision_weights.keys())[:10]):
    print(f"  {name}")

# Check the structure - what comes after vision_encoder.
prefixes = set()
for name in vision_weights.keys():
    parts = name.split(".")
    if len(parts) > 1:
        prefix = parts[1]
        prefixes.add(prefix)

print(f"\nVision encoder sub-modules: {sorted(prefixes)}")

# Create a dummy vision encoder to see expected parameter names
print("\n\nCreating dummy VisionTransformer to check expected parameter names...")

# Use the config from omnistral
vision_args = VisionEncoderArgs(
    hidden_size=1024,
    num_channels=3,
    image_size=1540,
    patch_size=14,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    rope_theta=10000.0,
    image_token_id=10,
    adapter_bias=True,
    spatial_merge_size=2,
    add_pre_mm_projector_layer_norm=True,
    mm_projector_id="patch_merge"
)

vision_encoder = VisionTransformer(vision_args)
expected_params = dict(vision_encoder.named_parameters())

print(f"\nExpected {len(expected_params)} parameters in VisionTransformer")
print("\nFirst 10 expected parameter names:")
for i, name in enumerate(sorted(expected_params.keys())[:10]):
    print(f"  {name}")

# Check for mismatches
checkpoint_names = {name.replace("vision_encoder.", "") for name in vision_weights.keys()}
expected_names = set(expected_params.keys())

missing_in_model = checkpoint_names - expected_names
missing_in_checkpoint = expected_names - checkpoint_names

if missing_in_model:
    print(f"\n\nWeights in checkpoint but not in model ({len(missing_in_model)}):")
    for name in sorted(missing_in_model)[:10]:
        print(f"  vision_encoder.{name}")
    if len(missing_in_model) > 10:
        print(f"  ... and {len(missing_in_model) - 10} more")

if missing_in_checkpoint:
    print(f"\n\nWeights expected by model but not in checkpoint ({len(missing_in_checkpoint)}):")
    for name in sorted(missing_in_checkpoint)[:10]:
        print(f"  {name}")
    if len(missing_in_checkpoint) > 10:
        print(f"  ... and {len(missing_in_checkpoint) - 10} more")
