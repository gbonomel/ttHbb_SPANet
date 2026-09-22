#!/bin/bash
BASE_DIR=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet
RUN_DIR_MC=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/training_multiplewp/nbjet_eff_btagsf/model_only_ttbar_2048/inference_spanet
RUN_DIR_DATA=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/training_multiplewp/nbjet_eff_btagsf/model_only_ttbar_2048/inference_spanet
EOS_DIR=/eos/user/g/gbonomel/tth/spanet/inference_pocketcoffea
MINORBK_2022_23=(
                 "TTHtoNon2B"
                 "TWminus" 
                 "TWplus" 
                 "TbarBLNuB_schannel" 
                 "TbarBQ_tchannel"
                 "TBbarLNuB_schannel" 
                 "TBbarQ_tchannel" 
                 "WJetsToLNu" 
                 "DYto2L-2Jets_MLL-10to50" 
                 "DYto2L-2Jets_MLL-50" 
                 "TTLL_ML_to50" 
                 "TTLL_ML_50" 
                 "TTNuNu" 
                 "TTLNu" 
                 "TTZ" 
                 "TTW"
                 "WW" 
                 "ZZ" 
                 "WZ"
                 )
MINORBK_2024=(
              "TTHtoNon2B"
              "TWminus" 
              "TWplus" 
              "WJetsToLNu" 
              "WJetsToLNu_4J_bin2J"
              "WJetsToLNu_4J_bin3J"
              "WJetsToLNu_4J_bin4J" 
              "DYto2L-PtLL_100_200_1J_PtLL_100_200_1J" 
              "DYto2L-PtLL_200_400_2J_PtLL_200_400_2J" 
              "DYto2L-PtLL_400_600_1J_PtLL_400_600_1J" 
              "DYto2L-PtLL_400_600_2J_PtLL_400_600_2J" 
              "DYto2L-PtLL_600_1J_PtLL_600_1J" 
              "DYto2L-PtLL_600_2J_PtLL_600_2J" 
              "TTLL_ML_to50" 
              "TTLL_ML_50"
              "TTNuNu" 
              "TTLNu" 
              "TTZ" 
              "TTW"
              "WW" 
              "ZZ" 
              "WZ"
              )
TTBAR=("TTTo2L2Nu" "TTToLNu2Q" "TTBBtoLNu2Q_4FS" "TTBBto2L2Nu_4FS")
SUBSAMPLES=("ttB" "ttC" "ttLF")
CAT=baseline

# Save ttHTobb_ttToSemiLep datasets to parquet
for YEAR in 2022_preEE 2022_postEE 2023_preBPix 2023_postBPix
do
    # ttHbb sample divided in semileptonic (containing genmatching) and non_semilep 
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_MC/output_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${YEAR}/TTH_Hto2B_semilep_${YEAR}.parquet --dataset TTH_Hto2B_${YEAR} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/TTH_Hto2B_${YEAR}/semilep/baseline --cat ${CAT}
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_MC/output_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${YEAR}/TTH_Hto2B_non_semilep_${YEAR}.parquet --dataset TTH_Hto2B_${YEAR} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/TTH_Hto2B_${YEAR}/non_semilep/baseline --cat ${CAT}
    for SAMPLE in "${MINORBK_2022_23[@]}"; do
        python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_MC/output_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${YEAR}/${SAMPLE}_${YEAR}.parquet --dataset ${SAMPLE}_${YEAR} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/${SAMPLE}_${YEAR}/baseline --cat ${CAT}
    done
    # ttbar semilepton, dilepton, 4FS and tthnon2b are split in subsamples ttLF, ttB, ttC
    for SAMPLE_TTBAR in "${TTBAR[@]}"; do
        for SUBSAMPLE in "${SUBSAMPLES[@]}"; do
            python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_MC/output_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${YEAR}/${SAMPLE_TTBAR}_${SUBSAMPLE}_${YEAR}.parquet --dataset ${SAMPLE_TTBAR}_${YEAR} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/${SAMPLE_TTBAR}_${YEAR}/${SUBSAMPLE}/baseline --cat ${CAT}
        done
    done
done

# separate 2024 because it has different variables (btagging algorithm)
for YEAR in 2024
do
    # ttHbb sample divided in semileptonic (containing genmatching) and non_semilep
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_MC/output_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${YEAR}/TTH_Hto2B_nonsemilep_${YEAR}.parquet --dataset TTH_Hto2B_2L2Nu_${YEAR} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/TTH_Hto2B_2L2Nu_${YEAR}/non_semilep/baseline --cat ${CAT}
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_MC/output_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${YEAR}/TTH_Hto2B_semilep_${YEAR}.parquet --dataset TTH_Hto2B_${YEAR} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/TTH_Hto2B_${YEAR}/semilep/baseline --cat ${CAT}
    for SAMPLE in "${MINORBK_2024[@]}"; do
        python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_MC/output_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${YEAR}/${SAMPLE}_${YEAR}.parquet --dataset ${SAMPLE}_${YEAR} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/${SAMPLE}_${YEAR}/baseline --cat ${CAT}
    done
     ttbar semilepton, dilepton, 4FS and tthnon2b are split in subsamples ttLF, ttB, ttC
    for SAMPLE_TTBAR in "${TTBAR[@]}"; do
        for SUBSAMPLE in "${SUBSAMPLES[@]}"; do
            python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_MC/output_${YEAR}.coffea -o $EOS_DIR/parquet_overall/${YEAR}/${SAMPLE_TTBAR}_${SUBSAMPLE}_${YEAR}.parquet --dataset ${SAMPLE_TTBAR}_${YEAR} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/${SAMPLE_TTBAR}_${YEAR}/${SUBSAMPLE}/baseline --cat ${CAT}
        done
    done
done

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

# process 2022-2023 data
for SAMPLE in "${DATA_EGAMMA[@]}"
do
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_DATA/output.coffea -o $EOS_DIR/parquet_overall/${YEAR}/${SAMPLE}.parquet --dataset ${SAMPLE} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/${SAMPLE}/DATA_EGamma/baseline/nominal --cat ${CAT}
done

# separate 2024 because it has different variables (btagging algorithm)
for SAMPLE in "${DATA_MUON[@]}"
do
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features_run3_semilep_dctr.yaml -i $RUN_DIR_DATA/output.coffea -o $EOS_DIR/parquet_overall/${YEAR}/${SAMPLE}.parquet --dataset ${SAMPLE} --ntuples $EOS_DIR/nutples/eff_nbj_only_ttbar_2048/${SAMPLE}/DATA_SingleMuon/baseline/nominal --cat ${CAT}
done