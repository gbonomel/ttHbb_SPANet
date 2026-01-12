#!/bin/bash

BASE_DIR=/eos/user/s/sspallac/spanet_export/parquet_overall/dileptonic

# --- DEFINITIONS ---
TTBAR=("TTTo2L2Nu" "TTToLNu2Q" "TTBBto2L2Nu_4FS" "TTBBto2L2Nu_v1"  "TTBBtoLNu2Q")
SUBSAMPLES=("ttB" "ttC" "ttLF")

SAMPLES_2022_23=(
    "TbarWplus"
    "TWminus" 
    "TTLL_MLL-50"
    "TTLL_MLL-4to50"
    "TTNuNu"
    "TTLNu"
    "TTZ-ZtoQQ"
    "TTH_Hto2B_v1_background"
    "TTH_Hto2B_v2_background"
)

SAMPLES_2024=(
    "TbarWplus"
    "TWminus" 
    #"TTLL_MLL-50"
    #"TTLL_MLL-4to50"
    #"TTNuNu"
    #"TTLNu"
    #"TTZ-ZtoQQ"
    "TTH_Hto2B_v1_background"
    #"TTH_Hto2B_v2_background"
)

# ==========================================
# LOOP FOR 2022-2023
# ==========================================
for year in "2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix";
do
    echo "Processing Year: $year"
    input_files=()
    
    # 1. Define Gen-matched (Signal) files
    genmatched_input_files=()
    sig_v1="/eos/user/s/sspallac/spanet_export/parquet_overall/dileptonic/TTH_Hto2B_v1_signal_${year}.parquet"
    sig_v2="/eos/user/s/sspallac/spanet_export/parquet_overall/dileptonic/TTH_Hto2B_v2_signal_${year}.parquet"
    
    # Check if signal files exist before adding
    if [ -f "$sig_v1" ]; then genmatched_input_files+=("$sig_v1"); fi
    if [ -f "$sig_v2" ]; then genmatched_input_files+=("$sig_v2"); fi

    # 2. Add Background Samples
    for sample in ${SAMPLES_2022_23[@]};
    do
        file="$BASE_DIR/${sample}_${year}.parquet"
        if [ -f "$file" ]; then
            input_files+=("$file")
        else
            echo "  [WARNING] Skipping missing file: $file"
        fi
    done

    # 3. Add TTBar Subsamples
    for sample_ttbar in ${TTBAR[@]}; do
        for subsample in "${SUBSAMPLES[@]}"; do
            file_subsample="$BASE_DIR/${sample_ttbar}_${subsample}_${year}.parquet"
            if [ -f "$file_subsample" ]; then
                input_files+=("$file_subsample")
            else
                echo "  [WARNING] Skipping missing file: $file_subsample"
            fi
        done
    done

    final_state=dileptonic
    
    # Run conversion for mixed/background files
    if [ ${#input_files[@]} -gt 0 ]; then
        output_file="/eos/user/s/sspallac/spanet_export/h5_files/dileptonic2/h5_file_${year}.h5"
        echo "  -> Running conversion for ${#input_files[@]} background files..."
        python /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py \
            --cfg /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/parameters/features_spanet_h5_run3_dileptonic.yaml \
            -i "${input_files[@]}" \
            -o "$output_file" \
            -f "$final_state"
    else
        echo "  [INFO] No background input files found for $year. Skipping command."
    fi
    
    # Run conversion for signal files
    if [ ${#genmatched_input_files[@]} -gt 0 ]; then
        genmatched_output_file="/eos/user/s/sspallac/spanet_export/h5_files/dileptonic2/signal_${year}.h5"
        echo "  -> Running conversion for signal files..."
        python /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py \
            --cfg /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/parameters/features_spanet_h5_run3_dileptonic.yaml \
            -i "${genmatched_input_files[@]}" \
            -o "$genmatched_output_file" \
            -f "$final_state"
    fi
done

# ==========================================
# LOOP FOR 2024
# ==========================================
for year in "2024";
do
    echo "Processing Year: $year"
    input_files=()
    
    # 1. Define Gen-matched (Signal) files
    genmatched_input_files=()
    sig_v1="/eos/user/s/sspallac/spanet_export/parquet_overall/dileptonic/TTH_Hto2B_v1_signal_${year}.parquet"
    
    if [ -f "$sig_v1" ]; then genmatched_input_files+=("$sig_v1"); fi

    # 2. Add Background Samples
    for sample in ${SAMPLES_2024[@]};
    do
        file="$BASE_DIR/${sample}_${year}.parquet"
        if [ -f "$file" ]; then
            input_files+=("$file")
        else
            echo "  [WARNING] Skipping missing file: $file"
        fi
    done

    # 3. Add TTBar Subsamples
    for sample_ttbar in ${TTBAR[@]}; do
        for subsample in "${SUBSAMPLES[@]}"; do
            file_subsample="$BASE_DIR/${sample_ttbar}_${subsample}_${year}.parquet"
            if [ -f "$file_subsample" ]; then
                input_files+=("$file_subsample")
            else
                echo "  [WARNING] Skipping missing file: $file_subsample"
            fi
        done
    done

    final_state=dileptonic
    
    # Run conversion for mixed/background files
    if [ ${#input_files[@]} -gt 0 ]; then
        output_file="/eos/user/s/sspallac/spanet_export/h5_files/dileptonic2/h5_file_${year}.h5"
        echo "  -> Running conversion for ${#input_files[@]} background files..."
        python /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py \
            --cfg /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/parameters/features_spanet_h5_run3_dileptonic.yaml \
            -i "${input_files[@]}" \
            -o "$output_file" \
            -f "$final_state"
    else
        echo "  [INFO] No background input files found for $year. Skipping command."
    fi
    
    # Run conversion for signal files
    if [ ${#genmatched_input_files[@]} -gt 0 ]; then
        genmatched_output_file="/eos/user/s/sspallac/spanet_export/h5_files/dileptonic2/signal_${year}.h5"
        echo "  -> Running conversion for signal files..."
        python /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py \
            --cfg /afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet/parameters/features_spanet_h5_run3_dileptonic.yaml \
            -i "${genmatched_input_files[@]}" \
            -o "$genmatched_output_file" \
            -f "$final_state"
    fi
done