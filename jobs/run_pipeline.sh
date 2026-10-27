#!/bin/bash
# Condor job submitted by scripts/run_pipeline.py --condor (in the PocketCoffea image): one parquet or h5 output.
# Arguments: <arguments of run_pipeline.py>
TTHBB_SPANET_DIR=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet

# Create venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

# Install ttHbb_SPANet in virtual environment
cd $TTHBB_SPANET_DIR
pip install -e .

cd $TTHBB_SPANET_DIR
python scripts/run_pipeline.py "$@"
