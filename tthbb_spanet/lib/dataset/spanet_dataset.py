import os
from enum import Enum
from collections import defaultdict

import numpy as np
import awkward as ak
import h5py

from .base import Dataset

# Inherited from https://github.com/Alexanders101/SPANet/blob/master/spanet/dataset/types.py
class SpecialKey(str, Enum):
    Mask = "MASK"
    Event = "EVENT"
    Inputs = "INPUTS"
    Targets = "TARGETS"
    Particle = "PARTICLE"
    Regressions = "REGRESSIONS"
    Permutations = "PERMUTATIONS"
    Classifications = "CLASSIFICATIONS"
    Embeddings = "EMBEDDINGS"
    Weights = "WEIGHTS"

class SPANetDataset(Dataset):
    def __init__(self, input_file, cfg, final_state, shuffle=True, reweigh=False, entrystop=None, has_data=False, fully_matched=False, batch_size=200000):
        self.fully_matched = fully_matched
        self.final_state = final_state
        # The input files are read and converted in batches of `batch_size` events, to limit the memory usage
        self.batch_size = batch_size
        super().__init__(input_file, cfg, shuffle=shuffle, reweigh=reweigh, entrystop=entrystop, has_data=has_data)

    def load_input(self):
        '''Only check the input files: the events are read batch by batch in `save`.'''
        self.check_inputs()
        print(f"Input: {len(self.input_files)} parquet files: ", self.input_files)
        return None

    def read_batches(self):
        '''Yield the events of the input files in batches of `batch_size` events, prepared as in `Dataset.load_input`
        (labels, weights, metadata) and with the fully matched selection applied if requested.'''
        import pyarrow as pa
        import pyarrow.parquet as pq
        for input_file in self.input_files:
            pf = pq.ParquetFile(input_file)
            for batch in pf.iter_batches(batch_size=self.batch_size, columns=self.columns_to_read(input_file)):
                df = self.prepare_input(ak.from_arrow(pa.Table.from_batches([batch])), input_file)
                if self.fully_matched:
                    df = self.select_fully_matched(df)
                yield df

    def count_events(self):
        '''Number of events of the input files (after the fully matched selection if requested).'''
        import pyarrow.parquet as pq
        if not self.fully_matched:
            return sum(pq.ParquetFile(f).metadata.num_rows for f in self.input_files)
        print("Counting the fully matched events...")
        return sum(len(df) for df in self.read_batches())

    def load_config(self):
        '''Load the config file with OmegaConf and read the input features for the SPANet training.'''
        super().load_config()
        self.input_features = self.cfg["input"]
        self.collection = self.cfg["collection"]
        self.targets = self.cfg["particles"]
        self.classification_targets = self.cfg["classification"]

    def columns_to_read(self, input_file):
        '''Only the collections used for the h5 file: reading all the columns of the parquet file (parton
        and generator level collections) takes ~3 times more memory.'''
        import pyarrow.parquet as pq
        available = pq.read_schema(input_file).names
        needed = {"event", "Event", "TargetLepTop", "TargetLepAntiTop", self.collection["Jet"]}
        if "Event" in self.input_features:
            needed.add("JetGood")   # ht
        needed.update(self.collection[obj] for obj in self.input_features if obj != "Event")
        columns = [c for c in available if c in needed]
        print(f"Reading {len(columns)}/{len(available)} columns: {columns}")
        return columns

    def select_fully_matched(self, df):
        '''Return only the fully matched events of `df`.'''
        mask_fullymatched = ak.sum(df[self.collection["Jet"]].matched == True, axis=1) >= 4
        df = df[mask_fullymatched]
        jets = df[self.collection["Jet"]]

        # 2. Higgs Check (Always 2 b-jets)
        jets_higgs = jets[jets.prov == 1]
        mask_match = ak.num(jets_higgs) == 2

        if self.final_state == "semileptonic":
            # Semileptonic: 2 Higgs + 2 Top(W->qq) + 1 Top(b) + 1 Top(b) = 6 jets usually
            # Your logic: 3 jets from W/HadTop
            jets_w_thadr = jets[(jets.prov == 5) | (jets.prov == 2)]
            mask_match = mask_match & (ak.num(jets_w_thadr) == 3)
            # 1 jet from LepTop
            jets_tlep = jets[jets.prov == 3]
            mask_match = mask_match & (ak.num(jets_tlep) == 1)

        elif self.final_state == "dileptonic":
            # Dileptonic: 2 Higgs + 1 Top(b) + 1 AntiTop(b) = 4 jets
            
            # Top 1 (b-jet, prov=2)
            jets_t1 = jets[jets.prov == 2]
            mask_match = mask_match & (ak.num(jets_t1) == 1)

            # Top 2 (b-jet, prov=3)
            jets_t2 = jets[jets.prov == 3]
            mask_match = mask_match & (ak.num(jets_t2) == 1)

        return df[mask_match]

    def check_output(self, output_file):
        '''Check the output file extension and if it already exists.'''
        # Check the output file extension
        filename, file_extension = os.path.splitext(output_file)
        if not file_extension == ".h5":
            raise ValueError(f"Output file {output_file} should be in .h5 format.")
        # Check if output file exists
        if os.path.exists(output_file):
            raise ValueError(f"Output file {output_file} already exists.")
        os.makedirs(os.path.abspath(os.path.dirname(output_file)), exist_ok=True)

    def create_groups(self):
        '''Create the groups in the h5 file.'''
        for target in self.targets:
            self.file.create_group(f"{SpecialKey.Targets.value}/{target}")
        for object in self.input_features.keys():
            self.file.create_group(f"{SpecialKey.Inputs.value}/{object}")
        for group in self.classification_targets:
            self.file.create_group(f"{SpecialKey.Classifications.value}/{group}")


    

    def create_classifications(self, df):
        '''Return the classification targets: {h5 path: (array, dtype)}.'''
        out = {}
        for group, targets in self.classification_targets.items():
            for target in targets:
                if group == SpecialKey.Event.value:
                    try:
                        values = df[target]
                    except:
                        raise Exception(f"Target {target} not found in the dataframe.")
                    out[f"{SpecialKey.Classifications.value}/{group}/{target}"] = (values, 'int64')
                else:
                    raise NotImplementedError
        return out

    def create_targets(self, df):
        '''Return the jet (and lepton) indices of the targets: {h5 path: (array, dtype)}.'''
        out = {}
        jets = df[self.collection["Jet"]]
        
        # Check if this is a signal sample with parton matching
        #if "prov" not in jets.fields:
        #    print(f"Warning: 'prov' field not found in jets. Skipping target creation for background sample.")
        #    print(f"Available fields: {jets.fields}")
        #    # For background samples, create dummy targets with all -1 (no matching)
        #    n_events = len(jets)
        #    
        #    for target in self.targets:
        #        if target == "h":
        #            self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b1", (n_events,), dtype='int64', data=np.full(n_events, -1))
        #            self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b2", (n_events,), dtype='int64', data=np.full(n_events, -1))
        #        elif target == "t1":
        #            self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/q1", (n_events,), dtype='int64', data=np.full(n_events, -1))
        #            self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/q2", (n_events,), dtype='int64', data=np.full(n_events, -1))
        #            self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/b", (n_events,), dtype='int64', data=np.full(n_events, -1))
        #        elif target == "t2":
        #            self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/b", (n_events,), dtype='int64', data=np.full(n_events, -1))
        #        else:
        #            raise NotImplementedError
        #    return
        
        # Original code for signal samples with parton matching
        indices = ak.local_index(jets)
        if self.final_state == "semileptonic":
            for target in self.targets:
            
                if target == "h":

                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:
                        mask = jets.prov == 1 # H->b1b2
                    # We select the local indices of jets matched with the Higgs
                    # The indices are padded with None such that there are 2 entries per event
                    # The None values are filled with -1 (a nan value).
                    indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                    index_b1 = indices_prov[:,0]
                    index_b2 = indices_prov[:,1]

                    out[f"{SpecialKey.Targets.value}/h/b1"] = (index_b1, 'int64')
                    out[f"{SpecialKey.Targets.value}/h/b2"] = (index_b2, 'int64')

                elif target == "t1":
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:
                        mask = jets.prov == 5 # W->q1q2 from t1
                    indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                    index_q1 = indices_prov[:,0]
                    index_q2 = indices_prov[:,1]
                    
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:
                        mask = jets.prov == 2 # t1->Wb
                    index_b_hadr = ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]

                    out[f"{SpecialKey.Targets.value}/t1/q1"] = (index_q1, 'int64')
                    out[f"{SpecialKey.Targets.value}/t1/q2"] = (index_q2, 'int64')
                    out[f"{SpecialKey.Targets.value}/t1/b"] = (index_b_hadr, 'int64')
                
                elif target == "t2":
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:
                        mask = jets.prov == 3 # t2->b
                    index_b_lep = ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]

                    out[f"{SpecialKey.Targets.value}/t2/b"] = (index_b_lep, 'int64')
                else:
                    raise NotImplementedError
        
        elif self.final_state == "dileptonic":

            leptons = df[self.collection["Lepton"]] 
            lepton_indices = ak.local_index(leptons)

            for target in self.targets:
            
                if target == "h":
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:
                        mask = jets.prov == 1 # H->b1b2
                    # We select the local indices of jets matched with the Higgs
                    # The indices are padded with None such that there are 2 entries per event
                    # The None values are filled with -1 (a nan value).
                    indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                    index_b1 = indices_prov[:,0]
                    index_b2 = indices_prov[:,1]

                    out[f"{SpecialKey.Targets.value}/h/b1"] = (index_b1, 'int64')
                    out[f"{SpecialKey.Targets.value}/h/b2"] = (index_b2, 'int64')
                
                elif target == "t1":
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:    
                        mask = jets.prov == 2
                    indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                    index_b1 = indices_prov[:,0]
                    out[f"{SpecialKey.Targets.value}/t1/b"] = (index_b1, 'int64')
                    if "Event" in df.fields and "TargetLepTop" in df["Event"].fields:
                        lep_idx = df["Event"]["TargetLepTop"]
                    
                    
                    elif "TargetLepTop" in df.fields:
                        lep_idx =  df["TargetLepTop"] 
                        #lep_idx = ak.fill_none(ak.pad_none(lep_idx, 1), -1)[:, 0]
#
                        #leptons = df[self.collection["Lepton"]]
                        #n_leptons = ak.num(leptons)
#
                        #is_valid = (lep_idx >= 0) & (lep_idx < n_leptons)
                        #lep_idx = ak.where(is_valid, lep_idx, -1)
#
                        #unique, counts = np.unique(ak.to_numpy(lep_idx), return_counts=True)
#
                        #for u, c in zip(unique, counts):
                        #    print(u, c)
                    else:
                        lep_idx = np.full(len(jets), -1, dtype='int64')                        
                    
                    out[f"{SpecialKey.Targets.value}/t1/l"] = (lep_idx, 'int64')

                elif target == 't2':
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:
                        mask = jets.prov == 3
                    indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                    index_b2 = indices_prov[:,0]
                    out[f"{SpecialKey.Targets.value}/t2/b"] = (index_b2, 'int64')

                    if "Event" in df.fields and "TargetLepAntiTop" in df["Event"].fields:
                        lep_idx = df["Event"]["TargetLepAntiTop"]
                    
                    elif "TargetLepAntiTop" in df.fields:
                        lep_idx = df["TargetLepAntiTop"] 
                    else:
                        lep_idx = np.full(len(jets), -1, dtype='int64')

                    out[f"{SpecialKey.Targets.value}/t2/l"] = (lep_idx, 'int64')
                else:
                    raise NotImplementedError
        return out

    def create_weights(self, df):
        '''Return the event weights: {h5 path: (array, dtype)}.'''
        return {f"{SpecialKey.Weights.value}/weight": (df.event.weight, 'float32')}

    def create_inputs(self, df):
        '''Return the input arrays: {h5 path: (array, dtype)}.'''
        features = self.get_object_features(df)
        out = {}
        for obj, feats in features.items():
            for feat, val in feats.items():
                if feat == "MASK":
                    dtype = 'bool'
                else:
                    dtype = 'float32'
                out[f"{SpecialKey.Inputs.value}/{obj}/{feat}"] = (val, dtype)
        return out

    def get_object_features(self, df):

        df_features = defaultdict(dict)
        for obj, features in self.input_features.items():

            features_dict = {}
            if obj == "Event":
                for feat in features:
                    if feat == "ht":
                        features_dict["ht"] = ak.sum(df["JetGood"]["pt"], axis=1)
                    else:
                        raise NotImplementedError
            else:
                collection = self.collection[obj]

                if (collection in df.fields):
                    objects = df[collection]
                    for feat in features:
                        if feat == "MASK":
                            continue
                        if feat in ["sin_phi", "cos_phi"]:
                            phi = objects["phi"]
                            if feat == "sin_phi":
                                values = np.sin(phi)
                            elif feat == "cos_phi":
                                values = np.cos(phi)
                        elif feat == "is_electron":
                            values = ak.values_astype(abs(objects["pdgId"]) == 11, int)
                        else:
                            values = objects[feat]
                        if objects.ndim == 1:
                            features_dict[feat] = ak.to_numpy(values)
                        elif objects.ndim == 2:
                            features_dict[feat] = ak.to_numpy(ak.fill_none(ak.pad_none(values, 16, clip=True), 0))
                        else:
                            raise NotImplementedError

                    if "MASK" in features:
                        if not "pt" in features:
                            raise NotImplementedError
                        features_dict["MASK"] = ~(features_dict["pt"] == 0)
                else:
                    raise ValueError(f"Collection {collection} not found in the parquet file.")
            df_features[obj] = features_dict
        return df_features

    def h5_tree(self, val, pre=''):
        items = len(val)
        for key, val in val.items():
            items -= 1
            if items == 0:
                # the last item
                if type(val) == h5py._hl.group.Group:
                    print(pre + '└── ' + key)
                    self.h5_tree(val, pre+'    ')
                else:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
            else:
                if type(val) == h5py._hl.group.Group:
                    print(pre + '├── ' + key)
                    self.h5_tree(val, pre+'│   ')
                else:
                    print(pre + '├── ' + key + ' (%d)' % len(val))

    def print(self):
        '''Print the h5 file tree.'''
        self.h5_tree(self.file)

    #def save(self, output_file):
    #    '''Save the h5 file for both the training and testing datasets.'''
    #    for dataset_type in ["train", "test"]:
    #        print(f"Processing dataset: {dataset_type} ({len(getattr(self, dataset_type))} events)")
#
    #        df = getattr(self, dataset_type)
    #        output_file_custom = output_file.replace(".h5", f"_{dataset_type}_{len(df)}.h5")
    #        self.check_output(output_file_custom)
#
    #        print("Creating output file: ", output_file_custom)
    #        self.file = h5py.File(output_file_custom, "w")
    #        self.create_groups()
    #        if not self.has_data:
    #            self.create_targets(df)
    #        self.create_classifications(df)
    #        self.create_inputs(df)
    #        self.create_weights(df)
    #        print(self.file)
    #        self.print()
    #        self.file.close()

    def plan_output(self, output_file, nevents, events_per_file):
        '''Return the output files [(name, dataset type, number of events)] and the output file index of each input event
        (-1 if the event is dropped by `entrystop`). As in `Dataset.train`/`test`, the events are shuffled, the first
        `entrystop` are kept and split in training and testing events, then in files of `events_per_file` events.'''
        nkeep = min(nevents, self.entrystop) if self.entrystop else nevents
        n_train = int((1-self.test_size)*nkeep)
        files = []
        for dataset_type, n in [("train", n_train), ("test", nkeep - n_train)]:
            if n == 0:
                print(f"Skipping {dataset_type} - empty dataset")
                continue
            n_files = (n + events_per_file - 1) // events_per_file
            for file_idx in range(n_files):
                size = min(events_per_file, n - file_idx*events_per_file)
                files.append((output_file.replace(".h5", f"_{dataset_type}_part{file_idx}_{size}.h5"), dataset_type, size))
        # Output file of each event: shuffling this array is equivalent to shuffling the events before splitting them
        index = np.full(nevents, -1, dtype=np.int16 if len(files) < 2**15 else np.int32)
        index[:nkeep] = np.repeat(np.arange(len(files)), [size for _, _, size in files])
        if self.shuffle:
            np.random.shuffle(index)
        return files, index

    def batch_arrays(self, df):
        '''Return the arrays of the h5 file for the events `df`: {h5 path: numpy array}.'''
        arrays = {}
        if not self.has_data:
            arrays.update(self.create_targets(df))
        arrays.update(self.create_classifications(df))
        arrays.update(self.create_inputs(df))
        arrays.update(self.create_weights(df))
        out = {}
        for name, (values, dtype) in arrays.items():
            values = ak.to_numpy(values) if isinstance(values, ak.Array) else np.asarray(values)
            out[name] = values.astype(dtype)
        return out

    def save(self, output_file, events_per_file=1000000):
        '''Save the h5 files for both training and testing datasets, splitting into smaller files.
        The input files are read in batches and each event is written to its output file, which is shuffled at the end:
        the memory usage does not depend on the number of events.

        Args:
            output_file: Base output file path
            events_per_file: Maximum number of events per file (default: 1M)
        '''
        nevents = self.count_events()
        files, index = self.plan_output(output_file, nevents, events_per_file)
        print(f"{nevents} input events, {np.sum(index >= 0)} kept: will create {len(files)} files with up to {events_per_file} events each")
        for name, _, _ in files:
            self.check_output(name)

        h5files, offsets, ievent = [], np.zeros(len(files), dtype=np.int64), 0
        try:
            for name, dataset_type, size in files:
                print(f"  Creating {name} ({size} events)")
                self.file = h5py.File(name, "w")
                self.create_groups()
                h5files.append(self.file)

            for df in self.read_batches():
                arrays = self.batch_arrays(df)
                batch_index = index[ievent:ievent + len(df)]
                ievent += len(df)
                for ifile in np.unique(batch_index[batch_index >= 0]):
                    sel = batch_index == ifile
                    n, start = int(np.sum(sel)), offsets[ifile]
                    for path, values in arrays.items():
                        if path not in h5files[ifile]:
                            h5files[ifile].create_dataset(path, (files[ifile][2],) + values.shape[1:], dtype=values.dtype)
                        ds = h5files[ifile][path]
                        if ds.shape[1:] != values.shape[1:]:
                            raise ValueError(f"{path}: shape {values.shape[1:]} differs from {ds.shape[1:]} of the previous events")
                        ds[start:start + n] = values[sel]
                    offsets[ifile] += n
                print(f"  {ievent}/{nevents} events processed", flush=True)

            if ievent != nevents:
                raise ValueError(f"{ievent} events read, {nevents} expected")
            for (name, dataset_type, size), f, n in zip(files, h5files, offsets):
                if n != size:
                    raise ValueError(f"{name}: {n} events written, {size} expected")
                if self.shuffle:
                    # The same permutation for all the datasets of the file, one dataset in memory at a time
                    perm = np.random.permutation(size)
                    datasets = []
                    f.visititems(lambda path, obj: datasets.append(obj) if isinstance(obj, h5py.Dataset) else None)
                    for ds in datasets:
                        ds[...] = ds[...][perm]
                f.close()
                print(f"  \u2713 Completed {name}")
        except BaseException:
            # Do not leave partial outputs behind
            for (name, _, _), f in zip(files, h5files):
                f.close()
                if os.path.exists(name):
                    os.remove(name)
            raise
        for dataset_type in ["train", "test"]:
            print(f"Finished {dataset_type}: created {sum(t == dataset_type for _, t, _ in files)} files")
