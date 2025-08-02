from safetensors import safe_open

def get_unique_key_patterns(file_path):
    with safe_open(file_path, framework="pt") as f:
        keys = f.keys()
        unique_key_patterns = set()
        for key in keys:
            parts = key.split('.')
            pattern_parts = []
            for part in parts:
                if part.isdigit():
                    pattern_parts.append('{}')
                else:
                    pattern_parts.append(part)
            key_pattern = '.'.join(pattern_parts)
            unique_key_patterns.add(key_pattern)
    return sorted(unique_key_patterns)

models = {
    "Omnistral": "/mnt/vast/home/avi/omni/voxtral_eyes/consolidated.safetensors",
    "Voxtral": "/mnt/vast/runs/mistral_release/instruct/250626_voxtral-small-250703/consolidated/consolidated.safetensors",
    "Pixtral": "/mnt/vast/runs/mistral_release/instruct/250530_mistral-small-2506/consolidated_bf16//consolidated.safetensors"
}

with open("keys.txt", "w") as f:
    for model_name, model_path in models.items():
        f.write(f"{model_name}:\n")
        try:
            unique_key_patterns = get_unique_key_patterns(model_path)
            for pattern in unique_key_patterns:
                f.write(f"  {pattern}\n")
        except Exception as e:
            f.write(f"  Error processing {model_name}: {e}\n")
        f.write("\n")
