"""Create the parquet _metadata of a folder of ntuple chunks saved by PocketCoffea.

The chunks must share the same schema, but PocketCoffea can save the same column with different types in
different chunks, e.g. the matched collections are float64 when some objects are not matched (padded) and
keep their original type (float32, int32, ...) in chunks where all the objects are matched. The chunks with
a different schema are rewritten in place with the columns cast to the common (promoted) type, then the
_metadata is created with awkward.
"""
import os
import sys
import glob
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
import awkward as ak

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("folder", help="Folder with the parquet chunks")
args = parser.parse_args()

files = sorted(glob.glob(os.path.join(args.folder, "*.parquet")))
if not files:
    sys.exit(f"ERROR: no parquet files in {args.folder}")
schemas = [pq.read_schema(f) for f in files]
# Numeric types are promoted to the widest one (e.g. float32, int64 -> float64)
common = pa.unify_schemas(schemas, promote_options="permissive")

for f, schema in zip(files, schemas):
    if schema.equals(common):
        continue
    diff = [n for n in common.names if n not in schema.names or not schema.field(n).equals(common.field(n))]
    print(f"Casting {len(diff)} columns of {os.path.basename(f)} to the common schema: {', '.join(diff)}")
    table = pq.read_table(f).cast(common)
    # Write next to the original (without the .parquet extension) and replace it only once complete
    tmp = f + ".tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, f)

ak.to_parquet.dataset(args.folder)
