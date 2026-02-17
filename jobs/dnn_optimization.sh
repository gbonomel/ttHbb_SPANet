#!/bin/bash
TTHBB_SPANET_DIR=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet
NUM_GPU=1

# Create venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

# Install ttHbb_SPANet in virtual environment
cd $TTHBB_SPANET_DIR
pip install -e .

# Install additional dependencies
pip install torch awkward vector mplhep scikit-learn omegaconf onnxruntime

# Arguments:
# $1 = input parquet file (DCTR input)
# $2 = model parquet file (SPANet output)
# $3 = event YAML file
# $4 = output directory
# $5 = number of epochs (optional, default 100)

cd $TTHBB_SPANET_DIR

if [ $# -eq 4 ]; then
    python scripts/dnn_optimization.py \
           --input $1 \
           --model $2 \
           --event $3 \
           --output $4
elif [ $# -eq 5 ]; then
    python scripts/dnn_optimization.py \
           --input $1 \
           --model $2 \
           --event $3 \
           --output $4 \
           --nepochs $5
fi
