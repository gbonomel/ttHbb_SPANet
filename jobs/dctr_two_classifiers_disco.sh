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
pip install torch awkward vector mplhep scikit-learn omegaconf pandas joblib

# Arguments:
# $1 = input parquet file (DCTR input)
# $2 = output directory
# $3 = YAML file with the DCTR training settings
# $4 = number of epochs (optional, overrides the training settings)

cd $TTHBB_SPANET_DIR

if [ $# -eq 3 ]; then
    python scripts/dctr_two_classifiers_disco.py \
           --input "$1" \
           --output "$2" \
           --config "$3"
elif [ $# -eq 4 ]; then
    python scripts/dctr_two_classifiers_disco.py \
           --input "$1" \
           --output "$2" \
           --config "$3" \
           --nepochs "$4"
fi
