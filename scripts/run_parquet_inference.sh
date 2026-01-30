#!/bin/bash

PARQUET_FILE=/eos/home-g/gbonomel/tth/dctr/training_datasets/for_inference/dataset_dctr_4j3b_withsignal_allyears_baseline.parquet
EVENT_FILE=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/dctr_lepton_veto/medium_model_spanet/spanet_output/version_0/event.yaml
ONNX_MODEL=/eos/home-g/gbonomel/tth/spanet/models/training_dec25/singlewp_tthmva_leptonveto_1024.onnx
OUTPUT_FOLDER=/eos/home-g/gbonomel/tth/spanet/models/singlewp_dctr_leptonveto_medium/spanet_evaluation 

# for gpu inference
python scripts/parquet_inference.py \
  --p "$PARQUET_FILE" \
  --event-file "$EVENT_FILE" \
  --onnx "$ONNX_MODEL" \
  --out-folder "$OUTPUT_FOLDER" \
  --workers 1 \
  --batch-size 8192 \
  --gpu \
  --save-scores \
  --qt

# for cpu inference
# python scripts/parquet_inference.py \
#   --p "$PARQUET_FILE" \
#   --event-file "$EVENT_FILE" \
#   --onnx "$ONNX_MODEL" \
#   --out-folder "$OUTPUT_FOLDER" \
#   --workers 8 \
#   --batch-size 4096 \
#   --save-scores \
#   --qt