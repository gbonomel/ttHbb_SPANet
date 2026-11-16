#!/bin/bash

DCTR_INPUT="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_5wps_multiplewp/no_triggerSF/dilep_channel/3j3b/dctr_inputs/dctr_datasets/baseline/22_23/dctr_dataset_3j3b_2L_baseline_22_23_train.parquet"
SAVE_FOLDER="/afs/cern.ch/work/g/gbonomel/higgs/ttHbb/dctr/dnn_optimization/new_baseline/spanet_v1_2L/dctr_22_23/v1_two_classifiers_disco"
# Training settings (CR cuts, network, DisCo lambdas, ...)
DCTR_CONFIG="/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/dctr_two_classifiers_disco_2L.yaml"
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
