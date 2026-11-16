import os
import argparse

import htcondor

from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument("-of", "--options_file", type=str, default=None,
                    help="JSON file with option overloads.")
parser.add_argument("-l", "--log_dir", type=str, default=None,
                    help="Output directory for the checkpoints and tensorboard logs. Default to current directory.")
parser.add_argument("-cf", "--checkpoint", type=str, default=None,
                    help="Optional checkpoint to load the training state from. "
                         "Fully restores model weights and optimizer state.")
# DNN optimization specific arguments
parser.add_argument("-i", "--input", type=str, default=None,
                    help="DCTR parquet input file (for dnn_optimization).")
parser.add_argument("-m", "--model_file", type=str, default=None,
                    help="SPANet model parquet file (for dnn_optimization).")
parser.add_argument("-e", "--event", type=str, default=None,
                    help="Event YAML file (for dnn_optimization).")
parser.add_argument("-o", "--output", type=str, default=None,
                    help="Output directory for plots and DCTR model (for dnn_optimization).")
parser.add_argument("--nepochs", type=int, default=100,
                    help="Number of epochs to train for (for dnn_optimization).")
# DCTR with two classifiers and DisCo specific arguments
parser.add_argument("--dctr-config", type=str, default=None,
                    help="YAML file with the DCTR training settings (for dctr_two_classifiers_disco).")
# Parquet inference specific arguments
parser.add_argument("--parquet-input", type=str, default=None,
                    help="Input parquet file for ONNX inference (for parquet_inference).")
parser.add_argument("--event-file", type=str, default=None,
                    help="Event YAML file for preprocessing (for parquet_inference).")
parser.add_argument("--onnx-model", type=str, default=None,
                    help="ONNX model file (for parquet_inference).")
parser.add_argument("--out-folder", type=str, default=None,
                    help="Output directory for inference parquet and optional transformer (for parquet_inference).")
parser.add_argument("--batch-size", type=int, default=8192,
                    help="Inference batch size (for parquet_inference).")
parser.add_argument("--workers", type=int, default=1,
                    help="Number of workers argument passed to parquet inference (for parquet_inference).")
parser.add_argument("--gpu", action="store_true",
                    help="Enable GPU provider for parquet inference.")
parser.add_argument("--qt", action="store_true",
                    help="Enable weighted quantile transformer save for parquet inference.")
parser.add_argument("--save-scores", action="store_true",
                    help="Enable output score parquet save for parquet inference.")
parser.add_argument("--max-events", type=int, default=None,
                    help="Optional cap on processed events (for parquet_inference).")
parser.add_argument('--dry', action="store_true")
parser.add_argument('--interactive', action="store_true")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--ncpu', type=int, default=3)
parser.add_argument('--request-memory', type=str, default=None,
                    help="Condor request_memory value, e.g. '24000' or '24 GB'.")
parser.add_argument("--good-gpus", action="store_true")
parser.add_argument("--args", nargs="+", type=str, help="additional args")
args = parser.parse_args()

interactive = args.interactive

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

cfg = OmegaConf.load(args.cfg)
basedir = cfg['path']
model = cfg['model']
job_flavour = cfg['job_flavour']
ngpu = cfg['ngpu']
ncpu = cfg['ncpu']
request_memory = cfg.get('request_memory', None)

# Override defaults by command line arguments
if args.ngpu != parser.get_default("ngpu"):
    ngpu = args.ngpu
if args.ncpu != parser.get_default("ncpu"):
    ncpu = args.ncpu
if args.request_memory is not None:
    request_memory = args.request_memory

print("Initializing job submission script...", end="\n\n")
sub = htcondor.Submit()

if interactive:
    sub['InteractiveJob'] = True

if model in ["jet_assignment", "classification", "classification_balance_weights", "classification_balance_weights_semileptonic", "test_validation_loss", "test_mdmm_loss"]:
    sub['Executable'] = f"{basedir}/jobs/{model}.sh"
    sub['arguments'] = f"{args.options_file} {args.log_dir}"
    sub['Output'] = f"{args.log_dir}/jobs/output/{model}-$(ClusterId).$(ProcId).out"
    sub['Error']  = f"{args.log_dir}/jobs/error/{model}-$(ClusterId).$(ProcId).err"
    sub['Log']    = f"{args.log_dir}/jobs/log/{model}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest"'
    sub['+JobFlavour'] = f'"{job_flavour}"'
elif model == "dnn_optimization":
    # Validate required arguments for dnn_optimization
    if not all([args.input, args.event, args.output]):
        raise ValueError("dnn_optimization requires --input, --event, and --output arguments")
    
    sub['Executable'] = f"{basedir}/jobs/{model}.sh"
    sub['arguments'] = f"{args.input} {args.model_file} {args.event} {args.output} {args.nepochs}"
    sub['Output'] = f"{args.output}/jobs/output/{model}-$(ClusterId).$(ProcId).out"
    sub['Error']  = f"{args.output}/jobs/error/{model}-$(ClusterId).$(ProcId).err"
    sub['Log']    = f"{args.output}/jobs/log/{model}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest"'
    sub['+JobFlavour'] = f'"{job_flavour}"'
elif model == "dctr_two_classifiers_disco":
    # Validate required arguments for dctr_two_classifiers_disco
    if not all([args.input, args.output, args.dctr_config]):
        raise ValueError("dctr_two_classifiers_disco requires --input, --output, and --dctr-config arguments")

    sub['Executable'] = f"{basedir}/jobs/{model}.sh"
    sub['arguments'] = f"{args.input} {args.output} {args.dctr_config} {args.nepochs}"
    sub['Output'] = f"{args.output}/jobs/output/{model}-$(ClusterId).$(ProcId).out"
    sub['Error']  = f"{args.output}/jobs/error/{model}-$(ClusterId).$(ProcId).err"
    sub['Log']    = f"{args.output}/jobs/log/{model}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest"'
    sub['+JobFlavour'] = f'"{job_flavour}"'
elif model == "parquet_inference":
    # Validate required arguments for parquet inference
    if not all([args.parquet_input, args.event_file, args.onnx_model, args.out_folder]):
        raise ValueError(
            "parquet_inference requires --parquet-input, --event-file, --onnx-model, and --out-folder arguments"
        )

    gpu_flag = 1 if args.gpu else 0
    qt_flag = 1 if args.qt else 0
    save_scores_flag = 1 if args.save_scores else 0
    max_events = -1 if args.max_events is None else args.max_events

    sub['Executable'] = f"{basedir}/jobs/{model}.sh"
    sub['arguments'] = (
        f"{args.parquet_input} {args.event_file} {args.onnx_model} {args.out_folder} "
        f"{args.batch_size} {args.workers} {gpu_flag} {qt_flag} {save_scores_flag} {max_events}"
    )
    sub['Output'] = f"{args.out_folder}/jobs/output/{model}-$(ClusterId).$(ProcId).out"
    sub['Error']  = f"{args.out_folder}/jobs/error/{model}-$(ClusterId).$(ProcId).err"
    sub['Log']    = f"{args.out_folder}/jobs/log/{model}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:latest"'
    sub['+JobFlavour'] = f'"{job_flavour}"'
else:
    raise ValueError(f"Model {model} not implemented")

# Load checkpoint
if args.checkpoint:
    sub['arguments'] += f" {args.checkpoint}"

# GPU and CPU requirements
sub['request_cpus'] = f"{args.ncpu}"
sub['request_gpus'] = f"{args.ngpu}"
if request_memory is not None:
    sub['request_memory'] = f"{request_memory}"

# Fix for CUDA issues on some nodes
sub['requirements'] = '!regexp("MIG", TARGET.Gpus_DeviceName)'

if args.good_gpus:
    sub['requirements'] = (
    '(regexp("A100", TARGET.GPUs_DeviceName) || '
    'regexp("V100", TARGET.GPUs_DeviceName) || '
    'regexp("H100", TARGET.GPUs_DeviceName)) && '
    '!regexp("MIG", TARGET.GPUs_DeviceName)'
)


print("Submission parameters:")
print(sub, end="\n\n")

print("Creating log folders...")
for folder in ["Output", "Error", "Log"]:
    os.makedirs(os.path.dirname(sub[folder]), exist_ok=True)
 
if not args.dry:
    print("Starting Condor scheduler...")
    client = htcondor.Schedd()
    result = client.submit(sub, count=1)

    print(f"Submitted {result.num_procs()} job(s) to {result.cluster()}")