import os
import argparse
import h5py
import numpy as np

# The merge is done out-of-core so that the memory usage does not scale with the total dataset size:
#   0. Optionally only a subset of the events is selected (--max-events), at random or per class (--balance-classes).
#   1. The output is split into blocks of --block-size events. Each selected event is assigned to a random
#      output block (exact random partition), and each input file is written dataset by dataset
#      into its slots in the output blocks.
#   2. Each output block is read back and shuffled in place.
# The combination of the two steps is equivalent to a uniform random permutation of all the events,
# while only one dataset of one file (step 1) or of one block (step 2) is kept in memory at a time.

# Number of events per class so that the total is num_entries and the classes are as balanced as possible:
# going from the smallest class, each class takes an equal share of the remaining events (or all its events
# if it has fewer). The largest class takes what is left, so the total is exactly num_entries.
def balanced_quotas(class_counts, num_entries):
    quotas = np.zeros_like(class_counts)
    remaining = num_entries
    for k, c in enumerate(np.argsort(class_counts)):
        quotas[c] = min(class_counts[c], remaining // (len(class_counts) - k))
        remaining -= quotas[c]
    return quotas

# Function to recursively collect the shape and dtype of all the datasets in a file
def read_layout(group, prefix=""):
    layout = {}
    for key, item in group.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            layout[name] = (item.shape, item.dtype)
        elif isinstance(item, h5py.Group):
            layout.update(read_layout(item, name))
    return layout

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", nargs="+", help="Input h5 files", required=True)
parser.add_argument("-o", "--output", help="Output file", required=True)
parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle the data")
parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
parser.add_argument("--block-size", type=int, default=5_000_000, help="Number of events per output block used for the shuffling (controls the memory usage)")
parser.add_argument("--seed", type=int, default=None, help="Random seed for the shuffling")
parser.add_argument("--max-events", type=int, default=None, help="Keep only this number of events, randomly selected among all the input files (the first ones if --no-shuffle)")
parser.add_argument("--balance-classes", type=str, default=None, metavar="DATASET",
                    help="With --max-events, select the same number of events per class of DATASET (e.g. CLASSIFICATIONS/EVENT/signal): "
                         "the classes with fewer events are taken entirely and the rest is shared among the other classes")
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

# Read the layout of all the files and check that they are compatible
layouts = []
for file_path in args.input:
    with h5py.File(file_path, 'r') as f:
        layouts.append(read_layout(f))
reference = layouts[0]
file_entries = []
for file_path, layout in zip(args.input, layouts):
    if layout.keys() != reference.keys():
        raise Exception(f"File {file_path} has different datasets than {args.input[0]}")
    for name, (shape, dtype) in layout.items():
        if shape[1:] != reference[name][0][1:] or dtype != reference[name][1]:
            raise Exception(f"Dataset {name} in {file_path} has shape {shape} and dtype {dtype}, expected {reference[name]}")
    n = layout["WEIGHTS/weight"][0][0]
    if any(shape[0] != n for shape, _ in layout.values()):
        raise Exception(f"Datasets in {file_path} have different number of entries")
    file_entries.append(n)
file_entries = np.array(file_entries, dtype=np.int64)
total_entries = int(file_entries.sum())
num_entries = total_entries if args.max_events is None else min(args.max_events, total_entries)
if args.balance_classes is not None and args.max_events is None:
    raise Exception("--balance-classes requires --max-events")
if args.balance_classes is not None and args.balance_classes not in reference:
    raise Exception(f"Dataset {args.balance_classes} not found in the input files")
print(f"Selecting {num_entries} events out of {total_entries} from {len(args.input)} files")
args.output = args.output.replace(".h5", f"_{num_entries}.h5")

# Check if the output file already exists
if not args.overwrite and os.path.exists(args.output):
    raise Exception(f"Output file {args.output} already exists")

# Select the events to keep
rng = np.random.default_rng(args.seed)
if args.balance_classes is not None:
    classes = []
    for file_path in args.input:
        with h5py.File(file_path, 'r') as f:
            classes.append(f[args.balance_classes][:])
    classes = np.concatenate(classes)
    class_values, class_index = np.unique(classes, return_inverse=True)
    del classes
    class_counts = np.bincount(class_index)
    quotas = balanced_quotas(class_counts, num_entries)
    selected = np.zeros(total_entries, dtype=bool)
    for c, (value, count, quota) in enumerate(zip(class_values, class_counts, quotas)):
        print(f"  class {value}: {quota}/{count} events")
        selected[rng.choice(np.flatnonzero(class_index == c), quota, replace=False)] = True
    del class_index
elif num_entries < total_entries:
    selected = np.zeros(total_entries, dtype=bool)
    if args.no_shuffle:
        selected[:num_entries] = True
    else:
        selected[rng.choice(total_entries, num_entries, replace=False)] = True
else:
    selected = np.ones(total_entries, dtype=bool)

# Assign each selected event to an output block: one block per input file with --no-shuffle (the input order
# is kept), otherwise blocks of --block-size events filled at random.
# The events not selected are assigned to an extra block (label num_blocks) that is not written.
if args.no_shuffle:
    num_blocks = len(args.input)
    labels = np.repeat(np.arange(num_blocks, dtype=np.int32), file_entries)
    labels[~selected] = num_blocks
else:
    num_blocks = max(1, int(np.ceil(num_entries / args.block_size)))
    block_labels = np.repeat(np.arange(num_blocks, dtype=np.int32), [len(b) for b in np.array_split(np.arange(num_entries), num_blocks)])
    labels = np.full(total_entries, num_blocks, dtype=np.int32)
    labels[selected] = rng.permutation(block_labels)
    del block_labels
del selected
block_sizes = np.bincount(labels, minlength=num_blocks + 1)[:num_blocks]
block_offsets = np.concatenate([[0], np.cumsum(block_sizes)[:-1]])
block_filled = np.zeros(num_blocks, dtype=np.int64)

with h5py.File(args.output, 'w') as fout:
    for name, (shape, dtype) in reference.items():
        fout.create_dataset(name, shape=(num_entries,) + shape[1:], dtype=dtype)

    file_offset = 0
    for i, file_path in enumerate(args.input):
        n = file_entries[i]
        file_labels = labels[file_offset:file_offset + n]
        counts = np.bincount(file_labels, minlength=num_blocks + 1)[:num_blocks]
        # Selected events grouped by output block, the not selected ones are at the end and dropped
        order = np.argsort(file_labels, kind="stable")[:counts.sum()]
        file_offset += n
        print(f"[{i+1}/{len(args.input)}] Writing {counts.sum()}/{n} events from {file_path}")
        if counts.sum() == 0:
            continue
        src_offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        dst_offsets = block_offsets + block_filled
        with h5py.File(file_path, 'r') as fin:
            for name in reference:
                data = fin[name][:][order]
                for j in np.nonzero(counts)[0]:
                    fout[name][dst_offsets[j]:dst_offsets[j] + counts[j]] = data[src_offsets[j]:src_offsets[j] + counts[j]]
        block_filled += counts

    if not np.array_equal(block_filled, block_sizes):
        raise Exception("Mismatch between the number of events written and the expected block sizes")

    # Shuffle the events inside each output block
    if not args.no_shuffle:
        for j in range(len(block_sizes)):
            print(f"Shuffling block {j+1}/{len(block_sizes)}")
            start, stop = block_offsets[j], block_offsets[j] + block_sizes[j]
            perm = rng.permutation(block_sizes[j])
            for name in reference:
                fout[name][start:stop] = fout[name][start:stop][perm]

msg = f"Data from {args.input} has been merged into {args.output}"
if not args.no_shuffle:
    msg = msg.replace("merged", "merged and shuffled")
print(msg)
