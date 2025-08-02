from safetensors import safe_open

# Open the safetensors file
file_path = "/mnt/vast/home/avi/omni/voxtral_eyes/consolidated.safetensors"
with safe_open(file_path, framework="pt") as f:
    # Get all the keys in the state dictionary
    keys = f.keys()

    # Extract unique key names by removing any numerical suffixes (e.g., layer indices)
    unique_keys = set()
    for key in keys:
        # Split the key into parts separated by dots
        parts = key.split('.')
        # Remove any numerical parts (assuming they represent layer indices)
        filtered_parts = [part for part in parts if not part.isdigit()]
        # Reconstruct the key without numerical parts
        unique_key = '.'.join(filtered_parts)
        unique_keys.add(unique_key)

    # Print the unique keys
    for unique_key in sorted(unique_keys):
        print(unique_key)
