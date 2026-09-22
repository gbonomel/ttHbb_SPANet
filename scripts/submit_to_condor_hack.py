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
parser.add_argument('--dry', action="store_true")
parser.add_argument('--interactive', action="store_true")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--ncpu', type=int, default=3)
parser.add_argument("--good-gpus", action="store_true")
parser.add_argument("--args", nargs="+", type=str, help="additional args")
args = parser.parse_args()

interactive = args.interactive

col = htcondor.Collector()
try:
    credd = htcondor.Credd()
except htcondor.HTCondorLocateError:
    credd_ad = col.locate(htcondor.DaemonTypes.Credd)
    credd = htcondor.Credd(credd_ad)
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

cfg = OmegaConf.load(args.cfg)
basedir = cfg['path']
model = cfg['model']
job_flavour = cfg['job_flavour']
ngpu = cfg['ngpu']
ncpu = cfg['ncpu']

# Override defaults by command line arguments
if args.ngpu != parser.get_default("ngpu"):
    ngpu = args.ngpu
if args.ncpu != parser.get_default("ncpu"):
    ncpu = args.ncpu

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
    if not all([args.input, args.model_file, args.event, args.output]):
        raise ValueError("dnn_optimization requires --input, --model_file, --event, and --output arguments")
    
    sub['Executable'] = f"{basedir}/jobs/{model}.sh"
    sub['arguments'] = f"{args.input} {args.model_file} {args.event} {args.output} {args.nepochs}"
    sub['Output'] = f"{args.output}/jobs/output/{model}-$(ClusterId).$(ProcId).out"
    sub['Error']  = f"{args.output}/jobs/error/{model}-$(ClusterId).$(ProcId).err"
    sub['Log']    = f"{args.output}/jobs/log/{model}-$(ClusterId).log"
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
    try:
        client = htcondor.Schedd()
    except htcondor.HTCondorLocateError:
        schedd_name = os.environ.get("_condor_SCHEDD_HOST") or os.environ.get("SCHEDD_NAME")
        if schedd_name:
            schedd_ad = col.locate(htcondor.DaemonTypes.Schedd, schedd_name)
        else:
            schedd_ads = col.query(htcondor.AdTypes.Schedd, projection=["Name", "MyAddress"])
            schedd_ad = None
            for ad in schedd_ads:
                name = ad.get("name")
                if name == "bigbird08.cern.ch": schedd_ad = ad
            if schedd_ad is None:
                raise RuntimeError(
                    "Unable to select a remote scheduler from HTCondor collector. "
                )
        #else:
        #    schedd_ads = col.query(htcondor.AdTypes.Schedd, projection=["Name", "MyAddress"])
        #    if not schedd_ads:
        #        raise RuntimeError(
        #            "Unable to locate any remote scheduler from HTCondor collector. "
        #            "Try exporting _condor_SCHEDD_HOST to a valid CERN schedd host."
        #        )
        #    schedd_ad = schedd_ads[0]
        print(f"Using remote schedd: {schedd_ad.get('Name', 'unknown')}")
        client = htcondor.Schedd(schedd_ad)
    result = client.submit(sub, count=1)

    print(f"Submitted {result.num_procs()} job(s) to {result.cluster()}")