BASE_DIR=/afs/cern.ch/work/g/gbonomel/higgs/SPANet_ETH

# Save ttHTobb_ttToSemiLep datasets to parquet
for YEAR in 2022_preEE 2022_postEE 2023_preBPix 2023_postBPix
do
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features.yaml -i /afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/generate_datasets/coffea_outputs/output_TTH_Hto2B_${YEAR}.coffea -o  /afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/generate_datasets/parquet/output_TTH_Hto2B_${YEAR}.parquet --dataset TTH_Hto2B_${YEAR} --ntuples /afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/generate_datasets/ntuples/TTH_Hto2B_${YEAR}/baseline
    python $BASE_DIR/tthbb_spanet/scripts/dataset/coffea_to_parquet.py --cfg $BASE_DIR/parameters/features.yaml -i /afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/generate_datasets/coffea_outputs/output_TTHtoNon2B_${YEAR}.coffea -o /afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/generate_datasets/parquet/output_TTHtoNon2B_${YEAR}.parquet --dataset TTHtoNon2B_${YEAR} --ntuples /afs/cern.ch/work/g/gbonomel/higgs/ttHbb/spanet/generate_datasets/ntuples/TTHtoNon2B_${YEAR}/TTToSemiLeptonic_tt+B/baseline
done
