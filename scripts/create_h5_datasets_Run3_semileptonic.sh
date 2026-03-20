#!/bin/bash
BASE_DIR=/afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet
EOS_DIR=/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_multiplewp/promptMVA_e_mu

SAMPLES_2022_23=(
        "TTH_Hto2B_nonsemilep"
        "TWminus" 
        "TWplus" 
        "TBbarQ_tchannel"
        "TbarBQ_tchannel"
        "TBbarLNuB_schannel"
        "TbarBLNuB_schannel"
        "TTNuNu" 
        "TTLNu" 
        "TTZ"
    ) 
    # WW, ZZ, WZ
    #"WJetsToLNu" 
    #"DYJetsToLL_10_50"
    #"DYJetsToLL_50" 
SAMPLES_2024=(
        "TTH_Hto2B_nonsemilep"
        "TWminus" 
        "TWplus"
        "TTLL_ML_to50" 
        "TTLL_ML_50" 
    )
    #"DYto2L-PtLL_100_200_1J_PtLL_100_200_1J" 
    #"DYto2L-PtLL_200_400_2J_PtLL_200_400_2J" 
    #"DYto2L-PtLL_400_600_1J_PtLL_400_600_1J" 
    #"DYto2L-PtLL_400_600_2J_PtLL_400_600_2J" 
    #"DYto2L-PtLL_600_1J_PtLL_600_1J" 
    #"DYto2L-PtLL_600_2J_PtLL_600_2J" 
TTBAR=("TTTo2L2Nu_ttC" "TTToLNu2Q_ttC" "TTBBtoLNu2Q_4FS_ttB" "TTBBto2L2Nu_4FS_ttB" "TTTo2L2Nu_ttLF" "TTToLNu2Q_ttLF") # 5FS for ttC and ttLF subsamples and 4FS for ttB

final_state=semileptonic

for year in "2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix";
do
    input_files=()
    genmatched_input_files=(
        "$EOS_DIR/parquet_overall/TTH_Hto2B_genmatching_${year}.parquet"
    )
    # Check each file and add to array if it exists
    for sample in ${SAMPLES_2022_23[@]};
    do
        file="$EOS_DIR/parquet_overall/${sample}_${year}.parquet"
        input_files+=("$file")
    done
    for sample_ttbar in ${TTBAR[@]}; do
        file_subsample="$EOS_DIR/parquet_overall/${sample_ttbar}_${year}.parquet"
        input_files+=("$file_subsample")
    done

    output_file="$EOS_DIR/h5_files/ttbb_tthbb_non_semilep_${year}.h5"
    command="python $BASE_DIR/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg $BASE_DIR/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${input_files[*]} -o $output_file -f ${final_state}"
    $command
    
    genmatched_output_file="$EOS_DIR/h5_files/tthbb_semilep_genmatched_${year}.h5"
    command_gen="python $BASE_DIR/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg $BASE_DIR/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${genmatched_input_files[*]} -o $genmatched_output_file -f ${final_state}"
    $command_gen
done

for year in "2024";
do
    input_files=()
    genmatched_input_files=(
        "$EOS_DIR/parquet_overall/TTH_Hto2B_genmatching_${year}.parquet"
    )
    # Check each file and add to array if it exists
    for sample in ${SAMPLES_2024[@]};
    do
        file="$EOS_DIR/parquet_overall/${sample}_${year}.parquet"
        input_files+=("$file")
    done
    for sample_ttbar in ${TTBAR[@]}; do
        file_subsample="$EOS_DIR/parquet_overall/${sample_ttbar}_${year}.parquet"
        input_files+=("$file_subsample")
    done

    output_file="$EOS_DIR/h5_files/ttbb_tthbb_non_semilep_${year}.h5"
    command="python $BASE_DIR/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg $BASE_DIR/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${input_files[*]} -o $output_file -f ${final_state}"
    $command
    
    genmatched_output_file="$EOS_DIR/h5_files/tthbb_semilep_genmatched_${year}.h5"
    command_gen="python $BASE_DIR/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg $BASE_DIR/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${genmatched_input_files[*]} -o $genmatched_output_file -f ${final_state}"
    $command_gen
done
