"""Pipeline from PocketCoffea output to SPANet inputs, configured by a single yaml file
(e.g. parameters/pipeline_run3_semileptonic.yaml).

Steps (in order), all the conversions are run one after the other:
  check     sanity checks of the configuration (always run)
  parquet   merge the ntuple chunks into one parquet file per dataset/subsample (always kept). The parquet
            _metadata of the chunks is created first if missing, and the conversion always reads through it.
  h5        convert the merged parquet files into h5 files (only with --h5 or --steps h5)

With --condor the sanity checks run locally and each parquet/h5 output becomes one HTCondor job (running this
script with --only <output>), with the memory set in the `condor` section of the configuration. The h5 files
need the merged parquet files: submit the h5 step once the parquet jobs are done.

The datasets to process are listed in the configuration, per group of years and per group of samples
(e.g. tthbb, ttbar, others), and shared between the parquet and h5 steps. Their PocketCoffea sample is
read from the merged coffea file (`merged_coffea`, PocketCoffea job outputs merged beforehand), their
subsamples (ttB, ttC, ttLF, ...) from the folder structure of the ntuples: a dataset split in subsamples
gives one parquet file per subsample.
Run it in the PocketCoffea container (it needs coffea and pyyaml), with a valid kerberos token to read EOS.

Examples:
  python3 scripts/run_pipeline.py -c parameters/pipeline_run3_semileptonic.yaml --dry-run
  python3 scripts/run_pipeline.py -c parameters/pipeline_run3_semileptonic.yaml --h5
  python3 scripts/run_pipeline.py -c parameters/pipeline_run3_semileptonic.yaml --steps h5 --years 2024
  python3 scripts/run_pipeline.py -c parameters/pipeline_run3_semileptonic.yaml --steps h5 --years 2024 --condor
"""
import os
import sys
import glob
import shutil
import argparse
import datetime
import subprocess
from dataclasses import dataclass

import yaml

ALL_STEPS = ["parquet", "h5"]
YEARS = ["2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]
# ["2016_PreVFP", "2016_PostVFP", "2017", "2018", "2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix", "2024"]


@dataclass
class Job:
    '''One merged parquet file: a dataset (or one of its subsamples) of a given year.'''
    dataset: str      # full PocketCoffea dataset name, e.g. MC_TTtoLNu2Q_2022_preEE
    base: str         # dataset name without year, e.g. MC_TTtoLNu2Q
    year: str
    group: str        # group of the dataset list in the configuration, e.g. ttbar
    sample: str       # PocketCoffea sample, e.g. MC_TT
    subsample: str    # e.g. ttB, None if the dataset is not split in subsamples
    coffea: str       # merged coffea file containing the dataset
    ntuples: str      # folder with the parquet chunks (where the _metadata is created)
    parquet: str      # merged output parquet file

    @property
    def label(self):
        '''Name without year, e.g. MC_TTtoLNu2Q_ttB'''
        return f"{self.base}_{self.subsample}" if self.subsample else self.base

    @property
    def name(self):
        return f"{self.label}_{self.year}"


class UniqueKeyLoader(yaml.SafeLoader):
    '''yaml loader raising an error on duplicated keys (OmegaConf would fail on them later).'''
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise yaml.constructor.ConstructorError(None, None, f"duplicated key `{key}`", key_node.start_mark)
            keys.add(key)
        return super().construct_mapping(node, deep)


def load_yaml(path):
    with open(path) as f:
        try:
            return yaml.load(f, Loader=UniqueKeyLoader)
        except yaml.YAMLError as e:
            sys.exit(f"ERROR: invalid yaml {path}:\n{e}")


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.cfg = load_yaml(args.config)
        self.base_dir = self.cfg["base_dir"]
        self.python = self.cfg.get("python", "python")
        self.cat = self.cfg["category"]
        self.var = self.cfg["variation"]
        all_years = [str(y) for g in self.cfg["datasets"].values() for y in g["years"]]
        self.years = [str(y) for y in args.years] if args.years else all_years
        self.parquet_dir = os.path.join(self.cfg["output"], "parquet_overall")
        self.h5_dir = os.path.join(self.cfg["output"], "h5_files")
        self.errors, self.warnings, self.failed = [], [], []
        self.condor_tasks = {step: [] for step in ALL_STEPS}
        self.read_merged_coffea()
        self.jobs = self.build_jobs()

    def path(self, p):
        return p if os.path.isabs(p) else os.path.join(self.base_dir, p)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    def read_merged_coffea(self):
        '''Read the datasets (with their sample and year) and the sample names of the cutflow
        (subsamples are named <sample>__<subsample>) from the merged coffea file.'''
        from coffea.util import load
        self.coffea = self.path(self.cfg["merged_coffea"])
        if not os.path.exists(self.coffea):
            sys.exit(f"ERROR: merged coffea file {self.coffea} not found: merge the PocketCoffea outputs first")
        print(f"Reading {self.coffea}")
        output = load(self.coffea)
        self.datasets = {d: (m["sample"], m["year"]) for d, m in output["datasets_metadata"]["by_dataset"].items()}
        if self.cat not in output["cutflow"]:
            sys.exit(f"ERROR: category `{self.cat}` not in {self.coffea} (categories: {', '.join(output['cutflow'])})")
        self.cutflow_samples = {s for d in output["cutflow"][self.cat].values() for s in d}

    def build_jobs(self):
        '''One job per dataset (or subsample) of the lists in `datasets` of the configuration.'''
        jobs = []
        for years_cfg in self.cfg["datasets"].values():
            groups = {k: v or [] for k, v in years_cfg.items() if k != "years"}
            for year in [str(y) for y in years_cfg["years"]]:
                if year not in self.years:
                    continue
                for group, bases in groups.items():
                    for base in bases:
                        jobs += self.dataset_jobs(base, year, group)
        return jobs

    def dataset_jobs(self, base, year, group):
        dataset = f"{base}_{year}"
        if dataset not in self.datasets:
            self.errors.append(f"Dataset {dataset} ({group}) not found in {self.coffea}")
            return []
        sample, _ = self.datasets[dataset]
        dataset_dir = os.path.join(self.cfg["ntuples"], dataset)
        try:
            os.listdir(dataset_dir)
        except FileNotFoundError:
            self.warnings.append(f"No ntuples for {dataset} (empty after selection or missing jobs)")
            return []
        except OSError as e:
            # e.g. EOS without a valid kerberos token: do not silently skip the dataset
            self.errors.append(f"Cannot read {dataset_dir}: {e}")
            return []
        # Datasets split in subsamples have one folder per subsample before the category
        if os.path.isdir(os.path.join(dataset_dir, self.cat)):
            folders = {None: dataset_dir}
        else:
            folders = {s: os.path.join(dataset_dir, s) for s in sorted(os.listdir(dataset_dir))
                       if os.path.isdir(os.path.join(dataset_dir, s, self.cat))}
        if not folders:
            self.warnings.append(f"No category `{self.cat}` in {dataset_dir}")
        jobs = []
        for subsample, folder in folders.items():
            job = Job(dataset, base, year, group, sample, subsample, self.coffea,
                      os.path.join(folder, self.cat, self.var), None)
            job.parquet = os.path.join(self.parquet_dir, year, f"{job.name}.parquet")
            jobs.append(job)
        return jobs

    def h5_files(self):
        '''Return {output h5: ([input parquet files], year, file configuration)} for the selected years.
        Each h5 file takes all the datasets of the groups in `groups` and the parquet files in `subsamples`.'''
        out = {}
        for name, fcfg in self.cfg["h5"]["files"].items():
            groups, subsamples = fcfg.get("groups") or [], fcfg.get("subsamples") or []
            for year in self.years:
                jobs = [job for job in self.jobs if job.year == year]
                inputs = [job.parquet for job in jobs if job.group in groups or job.label in subsamples]
                for label in set(subsamples) - {job.label for job in jobs}:
                    self.warnings.append(f"{name.format(year=year)}: no parquet file {label}_{year}")
                out[os.path.join(self.h5_dir, name.format(year=year) + ".h5")] = (inputs, year, fcfg)
        return out

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    def check(self):
        for y in self.years:
            if y not in YEARS:
                self.errors.append(f"Unknown year {y}")

        # Parquet features config: sample-specific features must refer to existing samples
        pfeat = load_yaml(self.path(self.cfg["parquet"]["cfg"]))
        for section in ("features", "features_pad"):
            for key in (pfeat[section].get("by_sample") or {}):
                if key not in self.cutflow_samples:
                    self.errors.append(f"{self.cfg['parquet']['cfg']}: {section}.by_sample.{key} does not match any "
                                       f"sample of the cutflow ({', '.join(sorted(self.cutflow_samples))})")

        # h5 config: each input file must match exactly one key of mapping_sample, present in mapping_encoding
        if "h5" in self.steps:
            hcfg = load_yaml(self.path(self.cfg["h5"]["cfg"]))
            mapping = hcfg["mapping_sample"]
            encodings = hcfg.get("mapping_encoding", {})
            groups = {k for g in self.cfg["datasets"].values() for k in g if k != "years"}
            for name, fcfg in self.cfg["h5"]["files"].items():
                for group in fcfg.get("groups") or []:
                    if group not in groups:
                        self.errors.append(f"h5 file {name}: unknown group {group} (groups: {', '.join(sorted(groups))})")
            for output, (inputs, _, _) in self.h5_files().items():
                if not inputs:
                    self.warnings.append(f"{output}: no input parquet selected")
                for f in inputs:
                    name = os.path.basename(f)
                    keys = [k for k in mapping if k in name]
                    if len(keys) != 1:
                        self.errors.append(f"{name}: matches {len(keys)} keys of mapping_sample {keys}, exactly 1 needed")
                        continue
                    for label, enc in encodings.items():
                        if keys[0] not in enc:
                            self.errors.append(f"{name}: key `{keys[0]}` missing in mapping_encoding.{label}")
            for label, enc in encodings.items():
                for key in enc:
                    if key not in mapping:
                        self.warnings.append(f"mapping_encoding.{label}.{key} is not in mapping_sample")

        print(f"Selected {len(self.jobs)} parquet outputs for years {self.years}")
        for w in self.warnings:
            print(f"WARNING: {w}")
        for e in self.errors:
            print(f"ERROR: {e}")
        return not self.errors

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------
    def run(self, label, cmd, outputs=()):
        '''Run a command in the foreground, with its output on the terminal. Return True on success.'''
        if self.args.dry_run:
            print("[dry-run]", " ".join(cmd))
            return True
        print(f"[{label}] {' '.join(cmd)}", flush=True)
        # Make tthbb_spanet importable by the scripts even if the package is not installed
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   PYTHONPATH=os.pathsep.join(filter(None, [self.base_dir, os.environ.get("PYTHONPATH")])))
        ret = subprocess.run(cmd, env=env).returncode
        if ret != 0:
            reason = f"killed by signal {-ret}, likely out of memory" if ret < 0 else f"exit code {ret}"
            self.failed.append(f"{label}: {reason}")
            # Do not leave partial outputs behind, otherwise they would be skipped at the next run
            for o in outputs:
                for p in [o] + (self.h5_parts(o) if o.endswith(".h5") else []):
                    if os.path.isfile(p):
                        os.remove(p)
        return ret == 0

    def selected(self, name):
        '''False if the output is not in --only (used by the condor jobs to run a single output).'''
        return not self.args.only or name in self.args.only

    def skip(self, path):
        return os.path.exists(path) and not self.args.overwrite

    @staticmethod
    def h5_parts(output):
        '''Files actually written by parquet_to_h5.py for `output`: <name>_{train,test}_part<i>_<nevents>.h5'''
        return glob.glob(output.replace(".h5", "_train_part*.h5")) + glob.glob(output.replace(".h5", "_test_part*.h5"))

    @staticmethod
    def metadata_is_stale(folder):
        '''True if the _metadata is missing or older than one of the parquet chunks (e.g. ntuples reproduced
        with the same file names): its schema and row-group offsets would not describe the new chunks.'''
        metadata = os.path.join(folder, "_metadata")
        if not os.path.exists(metadata):
            return True
        t = os.path.getmtime(metadata)
        return any(f.name.endswith(".parquet") and f.stat().st_mtime > t for f in os.scandir(folder))

    def create_metadata(self, job):
        '''Create the _metadata of the ntuple chunks of the job if missing or stale (or with --overwrite).
        ak.from_parquet only uses it when reading exactly the folder containing it, which is job.ntuples.'''
        if not self.args.overwrite and not self.metadata_is_stale(job.ntuples):
            return True
        print(f"[metadata] (re)creating {job.ntuples}/_metadata")
        return self.run(f"metadata {job.name}",
                        [self.python, "-c", "import sys, awkward as ak; ak.to_parquet.dataset(sys.argv[1])", job.ntuples])

    @staticmethod
    def parquet_is_outdated(job):
        '''True if the merged parquet is older than one of the ntuple chunks (ntuples reproduced after the conversion).'''
        t = os.path.getmtime(job.parquet)
        return any(f.name.endswith(".parquet") and f.stat().st_mtime > t for f in os.scandir(job.ntuples))

    def step_parquet(self):
        jobs = [job for job in self.jobs if self.selected(job.name) and
                (self.args.overwrite or not os.path.exists(job.parquet) or self.parquet_is_outdated(job))]
        print(f"[parquet] {len(jobs)} files to create in {self.parquet_dir}")
        if self.args.condor:
            self.condor_tasks["parquet"] = [(job.name, job.year, None) for job in jobs]
            return
        for i, job in enumerate(jobs, 1):
            print(f"----- [{i}/{len(jobs)}] {job.name}")
            if not self.create_metadata(job):
                self.failed.append(f"parquet {job.name}: not converted, the _metadata could not be created")
                continue
            if os.path.exists(job.parquet) and not self.args.dry_run:
                os.remove(job.parquet)   # --overwrite: coffea_to_parquet.py refuses existing outputs
            os.makedirs(os.path.dirname(job.parquet), exist_ok=True)
            self.run(f"parquet {job.name}",
                     [self.python, self.path("tthbb_spanet/scripts/dataset/coffea_to_parquet.py"),
                      "--cfg", self.path(self.cfg["parquet"]["cfg"]), "-i", job.coffea, "-o", job.parquet,
                      "--dataset", job.dataset, "--ntuples", job.ntuples, "--cat", self.cat,
                      "--sample", f"{job.sample}__{job.subsample}" if job.subsample else job.sample],
                     [job.parquet])

    def step_h5(self):
        os.makedirs(self.h5_dir, exist_ok=True)
        if self.args.condor and self.condor_tasks["parquet"]:
            print("[h5] parquet jobs submitted to condor: run again with --steps h5 --condor once they are done")
            return
        for output, (inputs, year, fcfg) in self.h5_files().items():
            name = os.path.basename(output)[:-len(".h5")]
            if not self.selected(name):
                continue
            parts = self.h5_parts(output)
            if parts and not self.args.overwrite:
                print(f"[h5] {output.replace('.h5', '_*part*.h5')} exists, skipping")
                continue
            if not inputs:
                self.failed.append(f"{output}: no input parquet files")
                continue
            missing = [f for f in inputs if not os.path.exists(f)]
            if missing and not self.args.dry_run:
                self.failed.append(f"{output}: missing inputs {missing}")
                continue
            if self.args.condor:
                # Memory can be set per h5 file (`request_memory` in h5.files), e.g. for the large signal files
                self.condor_tasks["h5"].append((name, year, fcfg.get("request_memory")))
                continue
            if not self.args.dry_run:
                for p in parts:   # --overwrite: parquet_to_h5.py refuses existing outputs
                    os.remove(p)
            self.run(f"h5 {os.path.basename(output)}",
                     [self.python, self.path("tthbb_spanet/scripts/dataset/parquet_to_h5.py"),
                      "--cfg", self.path(self.cfg["h5"]["cfg"]), "-i", *inputs, "-o", output,
                      "-f", self.cfg["h5"]["final_state"], *fcfg.get("args", [])],
                     [output])

    def submit_condor(self):
        '''Submit one condor job per output collected by the steps, each running this script with --only <output>.
        The submit file, the list of jobs and the job logs go in jobs/condor_pipeline/<date> (logs are not
        allowed on EOS).'''
        tasks = [(step, *t) for step in ALL_STEPS for t in self.condor_tasks[step]]
        if not tasks:
            print("[condor] nothing to submit")
            return
        ccfg = self.cfg.get("condor") or {}
        folder = self.path(os.path.join("jobs", "condor_pipeline", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        os.makedirs(folder, exist_ok=True)
        config = os.path.abspath(self.args.config)
        flags = [f for f in ("overwrite", "force") if getattr(self.args, f)]
        with open(os.path.join(folder, "jobs.txt"), "w") as f:
            for step, name, year, memory in tasks:
                scfg = ccfg.get(step) or {}
                args = ["-c", config, "--steps", step, "--years", year, "--only", name, *[f"--{x}" for x in flags]]
                # The last variable of `queue ... from` takes the rest of the line, i.e. all the arguments
                f.write(f"{name} {memory or scfg.get('request_memory', 4000)} {scfg.get('request_cpus', 1)} "
                        f"{scfg.get('job_flavour', 'workday')} {' '.join(args)}\n")
        sub = os.path.join(folder, "pipeline.sub")
        with open(sub, "w") as f:
            f.write(f"""executable     = {self.path("jobs/run_pipeline.sh")}
arguments      = $(args)
output         = {folder}/$(name).$(ClusterId).$(ProcId).out
error          = {folder}/$(name).$(ClusterId).$(ProcId).err
log            = {folder}/pipeline.$(ClusterId).log
request_memory = $(memory)
request_cpus   = $(cpus)
+JobFlavour    = "$(flavour)"
MY.SendCredential   = True
MY.SingularityImage = "{ccfg.get("image", "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-stable")}"
queue name, memory, cpus, flavour, args from {folder}/jobs.txt
""")
        print(f"[condor] {len(tasks)} jobs (memory in MB):")
        for step, name, _, memory in tasks:
            print(f"  {step:8s} {name}  {memory or (ccfg.get(step) or {}).get('request_memory', 4000)}")
        if self.args.dry_run:
            print(f"[dry-run] condor_submit {sub}")
        elif shutil.which("condor_submit"):
            if subprocess.run(["condor_submit", sub]).returncode != 0:
                self.failed.append(f"condor_submit {sub}")
        else:
            # Inside a container without condor_submit: use the python bindings (as submit_to_condor.py)
            try:
                import htcondor
                htcondor.Credd().add_user_cred(htcondor.CredTypes.Kerberos, None)
                with open(sub) as f:
                    description = htcondor.Submit(f.read())
                result = htcondor.Schedd().submit(description, itemdata=description.itemdata())
                print(f"[condor] submitted {result.num_procs()} jobs to cluster {result.cluster()}")
            except Exception as e:
                self.failed.append(f"condor submission of {sub} ({e}): run `condor_submit {sub}` outside the container")

    def execute(self):
        self.steps = self.args.steps or [s for s in ALL_STEPS if s != "h5" or self.args.h5]
        ok = self.check()
        if not ok and not self.args.force:
            sys.exit("Sanity checks failed, fix the errors above or run with --force")
        if self.args.check_only:
            return
        for step in ALL_STEPS:
            if step in self.steps:
                print(f"===== Step: {step} =====")
                getattr(self, f"step_{step}")()
        if self.args.condor:
            self.submit_condor()
        if self.failed:
            print("The following tasks failed:")
            for f in self.failed:
                print(f"  {f}")
            sys.exit(1)
        print("===== Done =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="Pipeline yaml configuration")
    parser.add_argument("--h5", action="store_true", help="Also create the h5 files")
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, help="Run only these steps (overrides --h5)")
    parser.add_argument("--years", nargs="+", help="Override the years of the configuration")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing outputs (by default they are skipped)")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands")
    parser.add_argument("--check-only", action="store_true", help="Only run the sanity checks")
    parser.add_argument("--force", action="store_true", help="Run even if the sanity checks fail")
    parser.add_argument("--condor", action="store_true",
                        help="Submit one HTCondor job per output instead of running locally (memory in `condor` of the config)")
    parser.add_argument("--only", nargs="+", help="Process only these outputs, e.g. MC_TTtoLNu2Q_ttB_2024 or tthbb_2024")
    Pipeline(parser.parse_args()).execute()
