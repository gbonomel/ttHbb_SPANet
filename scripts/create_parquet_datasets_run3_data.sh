#!/bin/bash
BASE_DIR=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet
RUN_DIR=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/training_multiplewp/promptMVA_e_mu
EOS_DIR=/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_multiplewp/promptMVA_e_mu

DATA_EGAMMA=(
    "DATA_EGamma_2022_postEE_EraE"
    "DATA_EGamma_2022_postEE_EraF"
    "DATA_EGamma_2022_postEE_EraG"
    "DATA_EGamma_2022_preEE_EraC"
    "DATA_EGamma_2022_preEE_EraD"
    "DATA_EGamma_2023_postBPix_EraD"
    "DATA_EGamma_2023_preBPix_EraCv1"
    "DATA_EGamma_2023_preBPix_EraCv2"
    "DATA_EGamma_2023_preBPix_EraCv3"
    "DATA_EGamma_2023_preBPix_EraCv4"
    "DATA_EGamma_2024_EraC"
    "DATA_EGamma_2024_EraD"
    "DATA_EGamma_2024_EraE"
    "DATA_EGamma_2024_EraF"
    "DATA_EGamma_2024_EraG"
    "DATA_EGamma_2024_EraH"
    "DATA_EGamma_2024_EraI"
)
DATA_MUON=(
    "DATA_SingleMuon_2022_postEE_EraE"
    "DATA_SingleMuon_2022_postEE_EraF"
    "DATA_SingleMuon_2022_postEE_EraG"
    "DATA_SingleMuon_2022_preEE_EraC"
    "DATA_SingleMuon_2022_preEE_EraD"
    "DATA_SingleMuon_2023_postBPix_EraD"
    "DATA_SingleMuon_2023_preBPix_EraCv1"
    "DATA_SingleMuon_2023_preBPix_EraCv2"
    "DATA_SingleMuon_2023_preBPix_EraCv3"
    "DATA_SingleMuon_2023_preBPix_EraCv4"
    "DATA_SingleMuon_2024_EraC"
    "DATA_SingleMuon_2024_EraD"
    "DATA_SingleMuon_2024_EraE"
    "DATA_SingleMuon_2024_EraF"
    "DATA_SingleMuon_2024_EraG"
    "DATA_SingleMuon_2024_EraH"
    "DATA_SingleMuon_2024_EraI"
)

CAT=baseline

# process 2022-2023 data
for SAMPLE in "${DATA_EGAMMA[@]}"
do
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep.yaml -i $RUN_DIR/output_coffea_data/output_${SAMPLE}.coffea -o $EOS_DIR/parquet_overall/${SAMPLE}.parquet --dataset ${SAMPLE} --ntuples $EOS_DIR/data/${SAMPLE}/DATA_EGamma/baseline/nominal --cat ${CAT}
done

# separate 2024 because it has different variables (btagging algorithm)
for SAMPLE in "${DATA_MUON[@]}"
do
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep.yaml -i $RUN_DIR/output_coffea_data/output_${SAMPLE}.coffea -o $EOS_DIR/parquet_overall/${SAMPLE}.parquet --dataset ${SAMPLE} --ntuples $EOS_DIR/data/${SAMPLE}/DATA_SingleMuon/baseline/nominal --cat ${CAT}
done