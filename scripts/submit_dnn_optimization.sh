#!/bin/bash

DCTR_INPUT="/eos/home-g/gbonomel/tth/dctr/training_datasets/btag_shapes/for_inference/dataset_dctr_4j3b_withsignal_allyears_baseline.parquet"
SPANET_MODEL="/eos/home-g/gbonomel/tth/spanet/models/singlewp_dctr_leptonveto_medium/spanet_evaluation/spanet_onnx_inference.parquet"
EVENT_FILE="/eos/home-g/gbonomel/tth/spanet/models/training_dec25/event.yaml"
SAVE_FOLDER="/eos/home-g/gbonomel/tth/dctr/plot_input_variables/test_dnn_optimization"
NEPOCHS=100

# Path to the config file for dnn_optimization
CFG="/eos/home-g/gbonomel/software/ttHbb_SPANet/jobs/config/dnn_optimization.yaml"

# Create output directory if it doesn't exist
mkdir -p $SAVE_FOLDER

# Submit to condor
python /eos/home-g/gbonomel/software/ttHbb_SPANet/scripts/submit_to_condor.py \
    --cfg $CFG \
    -i $DCTR_INPUT \
    -m $SPANET_MODEL \
    -e $EVENT_FILE \
    -o $SAVE_FOLDER \
    --nepochs $NEPOCHS \
    --good-gpus
