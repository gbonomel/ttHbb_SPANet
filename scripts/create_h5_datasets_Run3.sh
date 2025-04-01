#!/bin/bash

# Define the input files replacing tt+B from the inclusive sample to the ttbb one
years=("2022_preEE" "2022_postEE" "2023_preBPix" "2023_postBPix")
for year in ${years[@]};
do
    input_files=(
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTH_Hto2B_genmatching_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTH_Hto2B_nonsemilep_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTToLNu2Q_ttLF_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTToLNu2Q_ttB_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTToLNu2Q_ttC_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTTo2L2Nu_ttLF_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTTo2L2Nu_ttB_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTTo2L2Nu_ttC_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TWminus_${year}.parquet"
        #"/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall/TWplus_${year}.parquet"
        #"/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/WJetsToLNu_${year}.parquet"
        #"/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/ZZ_${year}.parquet "
        #"/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/WW_${year}.parquet"
        #"/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/WZ_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTLL_ML_to50_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTLL_ML_50_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTNuNu_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTLNu_${year}.parquet"
        "/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/parquet_overall_2/TTZ_${year}.parquet"
    )
    output_file="/eos/cms/store/group/phys_higgs/ttHbb/ntuples/non_semileptonic_tthbb_export/h5_files/tthbb_genmatching_ttbb_tthbb_non_semilep_${year}.h5"
    command="python /afs/cern.ch/work/g/gbonomel/higgs/ttHbb_SPANet/tthbb_spanet/scripts/dataset/parquet_to_h5.py --cfg parameters/features_spanet_h5_run3.yaml -i ${input_files[*]} -o $output_file"
    $command
done
