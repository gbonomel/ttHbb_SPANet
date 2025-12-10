#!/bin/bash

BASE_DIR=/eos/user/s/sspallac/spanet_export/parquet_overall/single_wp
years=("2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix")

for year in ${years[@]};
do
    input_files=()
    
    # List of all possible samples
    samples=(
        "TbarWplus"
        "TWminus" 
        "TTLL_MLL-50"
        "TTLL_MLL-4to50"
        "TTNuNu"
        "TTLNu"
        "TTZ-ZtoQQ"
        #"DYto2E_Bin-MLL-10to50" 
        #"DYto2E-2Jets_Bin-MLL-50"
        #"DYto2Mu-2Jets_Bin-MLL-50"
        #"DYto2Tau-2Jets_Bin-MLL-50" 
        #"WtoLNu+jets"
        #"WWto2L2Nu"
        #"WZto3LNu"
        #"WZto2L2Q"
        #"ZZto2L2Nu"
        #"WtoENu-4Jets"
        #"WtoMuNu-4Jets"
        #"WtoTauNu-4Jets"
        "TTToLNu2Q_ttB"
        "TTToLNu2Q_ttC"
        "TTToLNu2Q_ttLF"
        #"DATA_EGamma", 
        #"DATA_DoubleMuon", 
        #"DATA_MuonEG", 
        "TTTo2L2Nu_ttB"
        "TTTo2L2Nu_ttC"
        "TTTo2L2Nu_ttLF"
        "TTH_Hto2B_v1_signal"
        "TTH_Hto2B_v2_signal"
        "TTH_Hto2B_v1_background"
        "TTH_Hto2B_v2_background"
        "TTBBtoLNu2Q_ttB"
        "TTBBtoLNu2Q_ttC"
        "TTBBtoLNu2Q_ttLF"
        "TTBBto2L2Nu_v1_ttB"
        "TTBBto2L2Nu_v1_ttC"
        "TTBBto2L2Nu_v1_ttLF"
        "TTBBto2L2Nu_4FS_ttB"
        "TTBBto2L2Nu_4FS_ttC"
        "TTBBto2L2Nu_4FS_ttLF"
    )
    
    # Check each file and add to array if it exists
    for sample in ${samples[@]};
    do
        file="$BASE_DIR/${sample}_${year}.parquet"
        if [ -f "$file" ]; then
            input_files+=("$file")
        else
            echo "Warning: $file does not exist, skipping..."
        fi
    done
    
    # Only run if we have at least one input file
    if [ ${#input_files[@]} -gt 0 ]; then
        output_file="/eos/user/s/sspallac/spanet_export/h5_files/single_wp/h5_files_${year}.h5"
        echo "Processing year $year with ${#input_files[@]} files"
        python /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py \
            --cfg /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/parameters/features_spanet_h5_run3_dileptonic.yaml \
            -i ${input_files[@]} \
            -o $output_file
    else
        echo "No input files found for year $year, skipping..."
    fi
done