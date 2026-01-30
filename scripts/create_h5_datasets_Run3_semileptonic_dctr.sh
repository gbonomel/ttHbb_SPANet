#!/bin/bash

BASE_DIR=/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/dctr_lepton_veto/dctr_variables/parquet_overall

final_state=semileptonic

SAMPLES_2022_23=(
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
SAMPLES_2024=(
        "TWminus" 
        "TWplus"
        "TTLL_ML_to50" 
        "TTLL_ML_50" 
    )
TTBAR=("TTTo2L2Nu_ttC" "TTToLNu2Q_ttC" "TTBBtoLNu2Q_4FS_ttB" "TTTo2L2Nu_ttLF" "TTToLNu2Q_ttLF") # 5FS for ttC and ttLF subsamples and 4FS for ttB
    #"TTHtoNon2B"

for year in "2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix";
do
    input_files=()
    genmatched_input_files=(
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/dctr_lepton_veto/dctr_variables/parquet_overall/${year}/TTH_Hto2B_genmatching_${year}.parquet"
    )
    # Check each file and add to array if it exists
    for sample in ${SAMPLES_2022_23[@]};
    do
        file="$BASE_DIR/${year}/${sample}_${year}.parquet"
        input_files+=("$file")
    done
    for sample_ttbar in ${TTBAR[@]}; do
        file_subsample="$BASE_DIR/${year}/${sample_ttbar}_${year}.parquet"
        input_files+=("$file_subsample")
    done

    output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/dctr_lepton_veto/dctr_variables/h5_files/ttbb_tthbb_non_semilep_${year}.h5"
    command="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${input_files[*]} -o $output_file -f ${final_state}"
    $command
    
    genmatched_output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/dctr_lepton_veto/dctr_variables/h5_files/tthbb_semilep_genmatched_${year}.h5"
    command_gen="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${genmatched_input_files[*]} -o $genmatched_output_file -f ${final_state}"
    $command_gen
done

for year in "2024";
do
    input_files=()
    genmatched_input_files=(
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/dctr_lepton_veto/dctr_variables/parquet_overall/${year}/TTH_Hto2B_genmatching_${year}.parquet"
    )
    # Check each file and add to array if it exists
    for sample in ${SAMPLES_2024[@]};
    do
        file="$BASE_DIR/${year}/${sample}_${year}.parquet"
        input_files+=("$file")
    done
    for sample_ttbar in ${TTBAR[@]}; do
        file_subsample="$BASE_DIR/${year}/${sample_ttbar}_${year}.parquet"
        input_files+=("$file_subsample")
    done

    output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/dctr_lepton_veto/dctr_variables/h5_files/ttbb_tthbb_non_semilep_${year}.h5"
    command="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${input_files[*]} -o $output_file -f ${final_state}"
    $command
    
    genmatched_output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/ttHbb_export/training_fixedwp_dec25/training_no_toppt_var/fixed_wp/dctr_lepton_veto/dctr_variables/h5_files/tthbb_semilep_genmatched_${year}.h5"
    command_gen="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/parameters/features_spanet_h5_run3_semileptonic.yaml -i ${genmatched_input_files[*]} -o $genmatched_output_file -f ${final_state}"
    $command_gen
done
