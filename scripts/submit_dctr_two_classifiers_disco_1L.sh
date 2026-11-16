#!/bin/bash

DCTR_INPUT="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_5wps_multiplewp/no_triggerSF/semilep_channel/dctr_inputs/dctr_datasets/2024/dctr_dataset_5j3b_1L_2024_baseline.parquet"
SAVE_FOLDER="/afs/cern.ch/work/g/gbonomel/higgs/ttHbb/dctr/dnn_optimization/new_baseline/spanet_v0_1L/dctr_2024/v2_two_classifiers_disco"
# Training settings (CR cuts, network, DisCo lambdas, ...)
DCTR_CONFIG="/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/dctr_two_classifiers_disco_1L.yaml"
NEPOCHS=100

# Path to the config file for the condor job
CFG="/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/jobs/config/dctr_two_classifiers_disco.yaml"

# Create output directory if it doesn't exist
mkdir -p $SAVE_FOLDER

# Submit to condor
python3 /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/scripts/submit_to_condor.py \
    --cfg $CFG \
    -i $DCTR_INPUT \
    -o $SAVE_FOLDER \
    --dctr-config $DCTR_CONFIG \
    --nepochs $NEPOCHS \
    --good-gpus
