import os
import argparse
import h5py
import numpy as np

def get_structure(group, path=""):
    """Recursively get the structure of datasets in the h5 file."""
    structure = {}
    for key, item in group.items():
        current_path = f"{path}/{key}" if path else key
        if isinstance(item, h5py.Dataset):
            structure[current_path] = {
                'shape': item.shape[1:],  # Shape without first dimension (num_entries)
                'dtype': item.dtype
            }
        elif isinstance(item, h5py.Group):
            structure.update(get_structure(item, current_path))
    return structure

def get_dataset_size(group):
    """Get the number of entries (first dimension size) from any dataset."""
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            return item.shape[0]
        elif isinstance(item, h5py.Group):
            return get_dataset_size(item)
    return 0

def create_datasets(output_file, structure, total_size):
    """Create empty datasets in the output file with the correct total size."""
    for path, info in structure.items():
        full_shape = (total_size,) + info['shape']
        
        # Navigate to create groups if needed
        parts = path.split('/')
        group = output_file
        for part in parts[:-1]:
            if part not in group:
                group = group.create_group(part)
            else:
                group = group[part]
        
        # Create the dataset
        group.create_dataset(
            parts[-1],
            shape=full_shape,
            dtype=info['dtype'],
            chunks=True 
        )

def copy_data_recursive(input_group, output_file, structure, offset, input_path=""):
    """Recursively copy data from input to output at the given offset."""
    for key, item in input_group.items():
        current_path = f"{input_path}/{key}" if input_path else key
        
        if isinstance(item, h5py.Dataset):
            size = item.shape[0]
            output_file[current_path][offset:offset+size] = item[:]
        elif isinstance(item, h5py.Group):
            copy_data_recursive(item, output_file, structure, offset, current_path)

def shuffle_h5_inplace(file_path, num_entries, chunk_size=100000):
    """Shuffle the data in the h5 file in place using chunked operations."""
    print("Shuffling the data...")

    shuffled_indices = np.random.permutation(num_entries)
    
    # Get all dataset paths
    dataset_paths = []
    with h5py.File(file_path, 'r') as f:
        def collect_datasets(group, path=""):
            for key, item in group.items():
                current_path = f"{path}/{key}" if path else key
                if isinstance(item, h5py.Dataset):
                    dataset_paths.append(current_path)
                elif isinstance(item, h5py.Group):
                    collect_datasets(item, current_path)
        collect_datasets(f)
    
    # Shuffle each dataset
    with h5py.File(file_path, 'r+') as f:
        for idx, dataset_path in enumerate(dataset_paths):
            print(f"  Shuffling dataset {idx+1}/{len(dataset_paths)}: {dataset_path}")
            dataset = f[dataset_path]
            
            # Create temporary array to hold shuffled data
            temp_data = np.empty_like(dataset)
            
            # Read sequentially in chunks and write to shuffled positions
            for i in range(0, num_entries, chunk_size):
                end = min(i + chunk_size, num_entries)
                # Read original chunk
                chunk = dataset[i:end]
                # Write to shuffled positions
                temp_data[shuffled_indices[i:end]] = chunk
            
            # Write back the shuffled data
            dataset[:] = temp_data
            del temp_data  # Free memory



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", nargs="+", help="Input h5 files", required=True)
parser.add_argument("-o", "--output", help="Output file", required=True)
parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the data")
parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
args = parser.parse_args()

# Check if the input files exist
for f in args.input:
    if not os.path.exists(f):
        raise Exception(f"Input file {f} does not exist")

# Create output folder if it does not exist
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

# Check if the output file is in h5 format
if not args.output.endswith(".h5"):
    raise Exception("Output file must be in h5 format")

print("Analyzing input files...")
# Get structure and total size
structure = None
total_size = 0
for file_path in args.input:
    with h5py.File(file_path, 'r') as f:
        if structure is None:
            structure = get_structure(f)
        file_size = get_dataset_size(f)
        total_size += file_size
        print(f"  {os.path.basename(file_path)}: {file_size} entries")

print(f"\nTotal entries: {total_size}")

# Update output filename with total size
args.output = args.output.replace(".h5", f"_{total_size}.h5")

if not args.overwrite and os.path.exists(args.output):
    raise Exception(f"Output file {args.output} already exists (use --overwrite)")

print(f"\nCreating output file: {args.output}")
with h5py.File(args.output, 'w') as out_f:
    create_datasets(out_f, structure, total_size)
    
    print("Copying data from input files...")
    offset = 0
    for i, file_path in enumerate(args.input):
        print(f"  Processing file {i+1}/{len(args.input)}: {os.path.basename(file_path)}")
        with h5py.File(file_path, 'r') as in_f:
            file_size = get_dataset_size(in_f)
            copy_data_recursive(in_f, out_f, structure, offset)
            offset += file_size

# Shuffle if requested
if not args.no_shuffle:
    shuffle_h5_inplace(args.output, total_size)

msg = f"\nData from {len(args.input)} files has been merged into {args.output}"
if not args.no_shuffle:
    msg = msg.replace("merged", "merged and shuffled")
print(msg)
print(f"Total entries: {total_size}")