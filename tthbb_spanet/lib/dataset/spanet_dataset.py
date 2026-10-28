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
    def __init__(self, input_file, cfg, final_state, shuffle=True, reweigh=False, entrystop=None, has_data=False, fully_matched=False):
        super().__init__(input_file, cfg, shuffle=shuffle, reweigh=reweigh, entrystop=entrystop, has_data=has_data)
        self.fully_matched = fully_matched
        self.final_state = final_state
        if self.fully_matched:
            self.select_fully_matched()

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

    def select_fully_matched(self):
        '''Select only fully matched events.'''
        mask_fullymatched = ak.sum(self.dataset.df[self.collection["Jet"]].matched == True, axis=1) >= 4
        df = self.dataset.df[mask_fullymatched]
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
            print("Dileptonic final state")
            # Dileptonic: 2 Higgs + 1 Top(b) + 1 AntiTop(b) = 4 jets
            
            # Top 1 (b-jet, prov=2)
            jets_t1 = jets[jets.prov == 2]
            mask_match = mask_match & (ak.num(jets_t1) == 1)

            # Top 2 (b-jet, prov=3)
            jets_t2 = jets[jets.prov == 3]
            mask_match = mask_match & (ak.num(jets_t2) == 1)

        df = df[mask_match]
        print(f"Selected {len(df)} fully matched events ({self.final_state})")
        self.dataset.df = df

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
        '''Create the classification targets in the h5 file.'''
        for group, targets in self.classification_targets.items():
            for target in targets:
                if group == SpecialKey.Event.value:
                    try:
                        values = df[target]
                    except:
                        raise Exception(f"Target {target} not found in the dataframe.")
                    self.file.create_dataset(f"{SpecialKey.Classifications.value}/{group}/{target}", np.shape(values), dtype='int64', data=values)
                else:
                    raise NotImplementedError

    def create_targets(self, df):
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

                    self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b1", np.shape(index_b1), dtype='int64', data=index_b1)
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b2", np.shape(index_b2), dtype='int64', data=index_b2)

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

                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/q1", np.shape(index_q1), dtype='int64', data=index_q1)
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/q2", np.shape(index_q2), dtype='int64', data=index_q2)
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/b", np.shape(index_b_hadr), dtype='int64', data=index_b_hadr)
                
                elif target == "t2":
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:
                        mask = jets.prov == 3 # t2->b
                    index_b_lep = ak.fill_none(ak.pad_none(indices[mask], 1), -1)[:,0]

                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/b", np.shape(index_b_lep), dtype='int64', data=index_b_lep)
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

                    self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b1", np.shape(index_b1), dtype='int64', data=index_b1)
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/h/b2", np.shape(index_b2), dtype='int64', data=index_b2)
                
                elif target == "t1":
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:    
                        mask = jets.prov == 2
                    indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                    index_b1 = indices_prov[:,0]
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/b", np.shape(index_b1), dtype='int64', data=index_b1)
                    print(df.fields)
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
                    
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t1/l", np.shape(lep_idx), dtype='int64', data=lep_idx)

                elif target == 't2':
                    if "prov" not in jets.fields:
                        mask = ak.zeros_like(jets.pt, dtype=bool)
                    else:
                        mask = jets.prov == 3
                    indices_prov = ak.fill_none(ak.pad_none(indices[mask], 2), -1)

                    index_b2 = indices_prov[:,0]
                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/b", np.shape(index_b2), dtype='int64', data=index_b2)

                    if "Event" in df.fields and "TargetLepAntiTop" in df["Event"].fields:
                        lep_idx = df["Event"]["TargetLepAntiTop"]
                    
                    elif "TargetLepAntiTop" in df.fields:
                        lep_idx = df["TargetLepAntiTop"] 
                    else:
                        lep_idx = np.full(len(jets), -1, dtype='int64')

                    self.file.create_dataset(f"{SpecialKey.Targets.value}/t2/l", np.shape(lep_idx), dtype='int64', data=lep_idx)
                else:
                    raise NotImplementedError

    def create_weights(self, df):
        '''Create the weights in the h5 file.'''
        weights = df.event.weight
        print("Creating dataset: ", f"{SpecialKey.Weights.value}/weight")
        self.file.create_dataset(f"{SpecialKey.Weights.value}/weight", np.shape(weights), dtype='float32', data=weights)
        print(f"SpecialKey.Weights.value = {SpecialKey.Weights.value}")

    def create_inputs(self, df):
        '''Create the input arrays in the h5 file.'''
        features = self.get_object_features(df)

        for obj, feats in features.items():
            for feat, val in feats.items():
                if feat == "MASK":
                    dtype = 'bool'
                else:
                    dtype = 'float32'
                dataset_name = f"{SpecialKey.Inputs.value}/{obj}/{feat}"
                print("Creating dataset: ", dataset_name)
                ds = self.file.create_dataset(dataset_name, np.shape(val), dtype=dtype, data=val)

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

    def save(self, output_file, events_per_file=1000000):
        '''Save the h5 files for both training and testing datasets, splitting into smaller files.
        
        Args:
            output_file: Base output file path
            events_per_file: Maximum number of events per file (default: 2M)
        '''
        for dataset_type in ["train", "test"]:
            df = getattr(self, dataset_type)
            print(f"\nProcessing dataset: {dataset_type} ({len(df)} events)")
            
            if len(df) == 0:
                print(f"Skipping {dataset_type} - empty dataset")
                continue
            
            # Calculate number of files needed
            n_files = (len(df) + events_per_file - 1) // events_per_file
            print(f"Will create {n_files} files with up to {events_per_file} events each")
            
            # Split into multiple files
            for file_idx in range(n_files):
                start_idx = file_idx * events_per_file
                end_idx = min((file_idx + 1) * events_per_file, len(df))
                df_chunk = df[start_idx:end_idx]
                
                # Create filename with chunk info
                output_file_custom = output_file.replace(".h5", f"_{dataset_type}_part{file_idx}_{len(df_chunk)}.h5")
                self.check_output(output_file_custom)
                
                print(f"  File {file_idx+1}/{n_files}: Creating {output_file_custom} ({len(df_chunk)} events)")
                
                self.file = h5py.File(output_file_custom, "w")
                self.create_groups()
                if not self.has_data:
                    self.create_targets(df_chunk)
                    print("Targets created")
                self.create_classifications(df_chunk)
                self.create_inputs(df_chunk)
                self.create_weights(df_chunk)
                self.file.close()
                
                print(f"  ✓ Completed {output_file_custom}")
            
            print(f"Finished {dataset_type}: created {n_files} files")