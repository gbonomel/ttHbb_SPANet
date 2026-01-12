#!/bin/bash
SPANET_DIR=/afs/cern.ch/work/s/sspallac/private/project/SPANet_sspallac
TTHBB_SPANET_DIR=/afs/cern.ch/work/s/sspallac/private/project/ttHbb_SPANet
NUM_GPU=1

export CUDA_LAUNCH_BLOCKING=1
export TF_CPP_MIN_LOG_LEVEL=3
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create venv in local job dir
python -m venv myenv --system-site-packages
source myenv/bin/activate

# Install SPANet in virtual environment
cd $SPANET_DIR
pip install -e .

# Install ttHbb_SPANet in virtual environment
cd $TTHBB_SPANET_DIR
pip install -e .

# Install MDMM loss package
python -m pip install 'mdmm @ git+https://github.com/the-moliver/mdmm@master'

echo "Original CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# HTCondor sets this to "OCL0" for MIG slices, which crashes CUDA.
# We unset it so the driver sees the GPU mapped into the container by Singularity.
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
echo "Fixed CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


echo "**************** BREAKPOINT: GPU & ENV CHECK ****************"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo -e "\n1. Checking GPU via nvidia-smi:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "ERROR: nvidia-smi command not found!"
fi

echo -e "\n2. Checking Pip Packages (looking for torch versions):"
pip list | grep torch

echo -e "\n3. Checking PyTorch Visibility inside Python:"
python -c "import torch; print(f'Is CUDA available? {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Torch Version: {torch.__version__}')"
echo "*************************************************************"

# Launch training
if [ $# -eq 2 ]; then
    python -m spanet.train \
           --options_file $1 \
           --log_dir $2\
           --time_limit 07:00:00:00\
           --gpus $NUM_GPU 
elif [ $# -eq 3 ]; then
    python -m spanet.train \
           --options_file $1 \
           --log_dir $2\
           --checkpoint $3\
           --time_limit 07:00:00:00\
           --gpus $NUM_GPU
fi
