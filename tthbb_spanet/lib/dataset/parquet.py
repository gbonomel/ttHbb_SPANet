import os
import glob
from collections import defaultdict

import numba
import vector
vector.register_awkward()
vector.register_numba()

import numpy as np
import awkward as ak

from omegaconf import OmegaConf

from coffea.util import load
from coffea.processor.accumulator import column_accumulator
from coffea.processor import accumulate

class ParquetDataset:
    def __init__(self, input_file, output_file, cfg, dataset, cat="semilep_LHE", input_ntuples=None, sample=None, batch_size=200000):
        self.input_file = input_file
        self.output_file = output_file
        self.cfg = cfg
        self.dataset = dataset
        self.cat = cat
        self.input_ntuples = input_ntuples
        # Sample (or <sample>__<subsample>) of the ntuples in the cutflow. If None, it is guessed from the number of events
        self.sample = sample
        self.tolerance = 5
        # The ntuples are processed and written in batches of (at least) `batch_size` events, to limit the memory usage
        self.batch_size = batch_size

        self.load_config()
        self.load_input()
        self.check_output()
        self.normalize_genweights()
        self.load_features()
        if self.schema == "coffea":
            self.build_arrays_from_accumulator()
        elif self.schema != "parquet":
            raise ValueError(f"Schema {self.schema} not recognized.")
        # With the parquet schema the arrays are built batch by batch in save_parquet

    @property
    def nevents(self):
        if self.schema == "coffea":
            pass
        elif self.schema == "parquet":
            return self.n_ntuples

    def get_datasets(self, sample=None):
        if self.schema == "coffea":
            datasets = list(self.columns[sample].keys())
        elif self.schema == "parquet":
            datasets = [self.dataset]
        else:
            raise ValueError(f"Schema {self.schema} not recognized.")

        if len(datasets) == 0:
            raise ValueError(f"No dataset found for sample {sample}.")
        if len(datasets) > 1:
            raise ValueError(f"Multiple datasets found for sample {sample}.\nDatasets: {datasets}")

        return datasets

    @property
    def samples(self):
        if self.schema == "coffea":
            return list(self.columns.keys())
        elif self.schema == "parquet" and self.sample is not None:
            if self.sample not in self.cutflow[self.dataset]:
                raise ValueError(f"Sample {self.sample} not found in the cutflow of {self.dataset}: {list(self.cutflow[self.dataset].keys())}")
            nevt = self.cutflow[self.dataset][self.sample]
            nevt_val = nevt["nominal"] if isinstance(nevt, dict) else nevt
            if not np.isclose(nevt_val, self.nevents, atol=self.tolerance):
                raise ValueError(f"Sample {self.sample} has {nevt_val} events in the cutflow, but {self.nevents} events are in the ntuples.")
            return [self.sample]
        elif self.schema == "parquet":
            samples = []
            print(f"DEBUG: Cutflow keys for dataset '{self.dataset}':")
            for sample, nevt in self.cutflow[self.dataset].items():
                nevt_val = nevt["nominal"] if isinstance(nevt, dict) else nevt
                print(f"  - {sample}: {nevt_val} events")
                if self.tolerance != None:                    
                    if np.isclose(nevt_val, self.nevents, atol=self.tolerance):
                        samples.append(sample)
                else:
                    if nevt == self.nevents:
                        samples.append(sample)
            print(f'Samples: {samples}')
            #breakpoint()
            if len(samples) == 0:
                raise ValueError(f"No sample found with {self.nevents} events.")
            return samples

    def is_mc(self, sample):
        if sample.startswith("DATA"):
            return False
        else:
            return True

    def load_input(self):
        '''Load the input file and check if the event category `cat` is present.'''
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file {self.input_file} does not exist.")
        else:
            print(f"Reading input file: {self.input_file}")
            self.df = load(self.input_file)
            self.columns = self.df["columns"]
            if not self.cat in self.df["cutflow"].keys():
                raise ValueError(f"Event category `{self.cat}` not found in the input file.")
            self.sum_genweights = self.df["sum_genweights"]
            self.cutflow = self.df["cutflow"][self.cat]
            
        if self.input_ntuples:
            self.schema = "parquet"
            print("Reading input ntuples: ", self.input_ntuples)
            # Only the footers are read here, the events are read batch by batch in save_parquet
            import pyarrow.parquet as pq
            files = sorted(glob.glob(os.path.join(self.input_ntuples, "*.parquet")))
            nrows = [pq.ParquetFile(f).metadata.num_rows for f in files]
            self.parquet_files = [f for f, n in zip(files, nrows) if n > 0]
            self.n_ntuples = sum(nrows)
            self.genweight_norm = None
            self.columns_ntuples = self.columns_to_read()
            print(f"Number of events: {self.nevents} in {len(self.parquet_files)} non-empty chunks")
            if self.nevents == 0:
                raise ValueError(f"No events in the ntuples {self.input_ntuples}")
        else:
            self.schema = "coffea"

        # Build the array dictionary and the zipped dictionary for each sample
        self.array_dict = {s : defaultdict(dict) for s in self.samples}
        self.zipped_dict = {s : defaultdict(dict) for s in self.samples}
        self.features_dict = {s : defaultdict(dict) for s in self.samples}
        self.features_pad_dict = {s : defaultdict(dict) for s in self.samples}

    def columns_to_read(self):
        '''Columns of the ntuples needed for the features of `self.sample`, to limit the memory usage.
        Return None (all columns) if the sample is not given or the schema of the ntuples is not available.'''
        schema_file = os.path.join(self.input_ntuples, "_common_metadata")
        if self.sample is None or not os.path.exists(schema_file):
            return None
        import pyarrow.parquet as pq
        available = pq.read_schema(schema_file).names
        features = dict(self.features["common"])
        features.update(self.features["by_sample"].get(self.sample, {}))
        needed = {"weight"}
        for collection, variables in features.items():
            for key_coffea in variables.values():
                needed.add(f"{collection}_{key_coffea}")
                if (collection == "JetGoodMatched") & (key_coffea == "provenance"):
                    needed.add("PartonMatched_provenance")
        columns = [c for c in available if c in needed]
        print(f"Reading {len(columns)}/{len(available)} columns of the ntuples")
        return columns

    def load_config(self):
        '''Load the features, features_pad, awkward_collections and matched_collections_dict
        dictionaries with OmegaConf.'''
        print(f"Reading configuration file: {self.cfg}")
        self.cfg = OmegaConf.load(self.cfg)
        self.features = self.cfg["features"]
        self.features_pad = self.cfg["features_pad"]
        self.awkward_collections = self.cfg["awkward_collections"]
        self.matched_collections_dict = self.cfg["matched_collections"]

    def check_output(self):
        # Check if the output file is already existing
        if os.path.exists(self.output_file):
            raise ValueError(f"Output file {self.output_file} already exists.")
        # Check that the format of the output file is `.parquet`
        if not self.output_file.endswith(".parquet"):
            raise ValueError(f"Output file {self.output_file} should have the `.parquet` extension.")
        # Create the output directory if it does not exist
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)

    def get_weight(self, sample, dataset):
        if self.schema == "coffea":
            weight = self.columns[sample][dataset][self.cat]["weight"].value
        return weight

    @staticmethod
    def get_collection_dim(collection_dict):
        '''Get the dimension of the collection from the dictionary of arrays.'''
        ndims = []
        for key, values in collection_dict.items():
            ndims.append(values.ndim)
        ndims = set(ndims)
        if len(ndims) > 1:
            raise ValueError(f"The collection dimension cannot be determined as there are columns of different dimensions: {ndims}")
        return list(ndims)[0]

    def normalize_genweights(self):
        '''Since the array `weight` is filled on the fly with the weight associated with the event, 
        it does not take into account the overall scaling by the sum of genweights (`sum_genweights`).
        In order to correct for this, we have to scale by hand the `weight` array dividing by the sum of genweights.'''

        print("Normalizing genweights...")
        for sample in self.samples:
            if self.is_mc(sample):
                for dataset in self.get_datasets(sample):
                    if self.schema == "coffea":
                        weight = self.get_weight(sample, dataset)
                        weight_new = column_accumulator(weight / self.sum_genweights[dataset])
                        self.columns[sample][dataset][self.cat]["weight"] = weight_new
                    elif self.schema == "parquet":
                        # Applied to each batch of events in read_batches
                        self.genweight_norm = self.sum_genweights[dataset]

    def load_features(self):
        '''Load the features dictionary with common features and sample-specific features.'''
        
        print("Loading features...")
        # Compose the features dictionary with common features and sample-specific features
        for sample in self.samples:
            self.features_dict[sample] = self.features["common"].copy()
            if sample in self.features["by_sample"].keys():
                self.features_dict[sample].update(self.features["by_sample"][sample])

            # Compose the dictionary of features to pad
            self.features_pad_dict[sample] = self.features_pad["common"].copy()
            if sample in self.features_pad["by_sample"].keys():
                self.features_pad_dict[sample].update(self.features_pad["by_sample"][sample])

            # Create a default dictionary of dictionaries to store the arrays
            self.array_dict[sample] = {k : defaultdict(dict) for k in self.features_dict[sample].keys()}

    def read_batches(self):
        '''Yield the events of the ntuples in batches of at least `batch_size` events (whole chunks are read),
        with the weight normalized by the sum of genweights.'''
        import pyarrow as pa
        import pyarrow.parquet as pq
        tables, nrows = [], 0
        for i, f in enumerate(self.parquet_files):
            table = pq.read_table(f, columns=self.columns_ntuples)
            tables.append(table)
            nrows += table.num_rows
            if nrows >= self.batch_size or i == len(self.parquet_files) - 1:
                # The chunks share the same schema (see create_parquet_metadata.py)
                events = ak.from_arrow(pa.concat_tables(tables))
                tables, nrows = [], 0
                if self.genweight_norm is not None:
                    events = ak.with_field(events, events.weight / self.genweight_norm, "weight")
                yield events

    def build_arrays_from_ntuples(self, events, sample, verbose=False):
        '''Build the Momentum4D arrays for the jets, partons, leptons, met and higgs of a batch of `events`,
        read from unflattened ntuples. Return the dictionary of collections.'''

        array_dict = {k : defaultdict(dict) for k in self.features_dict[sample].keys()}
        zipped_dict = {}

        # Build the array_dict from the unflattened ntuples read from parquet
        for collection, variable in self.features_dict[sample].items():
            for key_feature, key_coffea in variable.items():
                if (collection == "JetGoodMatched") & (key_coffea == "provenance"):
                    try:
                        array_dict[collection][key_feature] = events[f"JetGoodMatched_provenance"]
                    except:
                        array_dict[collection][key_feature] = events[f"PartonMatched_provenance"]
                else:
                    values = events[f"{collection}_{key_coffea}"]
                    if (collection not in self.awkward_collections) & (values.ndim > 1):
                        values = ak.flatten(values)
                    array_dict[collection][key_feature] = values

            # Add padded features to the array, according to the features dictionary
            if collection in self.features_pad_dict[sample].keys():
                for key_feature, value in self.features_pad_dict[sample][collection].items():
                    array_dict[collection][key_feature] = value * ak.ones_like(array_dict[collection]["pt"])

        # The awkward arrays are zipped together to form the Momentum4D arrays.
        # If the collection is not a Momentum4D array, it is zipped as it is.
        # Here we assume that the ntuples are unflattened, therefore we don't need to unflatten them before zipping.
        for collection in array_dict.keys():
            ndim = self.get_collection_dim(array_dict[collection])
            if (collection not in self.awkward_collections) & (ndim > 1):
                # We flatten the collections that should not be saved as awkward collections after zipping
                zipped_dict[collection] = ak.flatten(ak.zip(array_dict[collection], with_name='Momentum4D'))
            else:
                zipped_dict[collection] = ak.zip(array_dict[collection], with_name='Momentum4D')
            if verbose:
                print(f"Collection: {collection}")
                print("Fields:", zipped_dict[collection].fields)

        for collection in list(zipped_dict.keys()):
            # Pad the matched collections with None if there is no matching
            if collection in self.matched_collections_dict.keys():
                matched_collection = self.matched_collections_dict[collection]

                if matched_collection not in zipped_dict:
                    if verbose:
                        print(f"WARNING: Matched collection '{matched_collection}' not found for sample '{sample}'. Skipping.")
                    continue

                masked_arrays = ak.mask(zipped_dict[matched_collection], zipped_dict[matched_collection].pt==-999, None)
                zipped_dict[matched_collection] = masked_arrays
                # Add the matched flag and the provenance to the matched jets
                if collection == "JetGood":
                    is_matched = ~ak.is_none(masked_arrays, axis=1)
                    zipped_dict[collection] = ak.with_field(zipped_dict[collection], is_matched, "matched")
                    zipped_dict[collection] = ak.with_field(zipped_dict[collection], ak.fill_none(masked_arrays.prov, -1), "prov")

        # Add the remaining keys to the zipped dictionary
        if self.is_mc(sample):
            zipped_dict["event"] = ak.zip({"weight" : events.weight})

        for collection, arr in zipped_dict.items():
            if len(arr) != len(events):
                raise ValueError(f"Collection {collection} has {len(arr)} entries for {len(events)} events: "
                                 f"add it to `awkward_collections` if it has a variable number of objects per event")
        return zipped_dict

    def build_arrays_from_accumulator(self):
        '''Build the Momentum4D arrays for the jets, partons, leptons, met and higgs.'''
        
        print("Samples: ", self.samples)

        for sample in self.samples:
            datasets = self.get_datasets(sample)
            print("Datasets: ", datasets)

            # Accumulate ntuples from different data-taking eras
            cs = accumulate([self.columns[sample][dataset][self.cat] for dataset in datasets])

            # In order to get the numpy array from the column_accumulator, we have to access the `value` attribute.
            for collection, variables in self.features_dict[sample].items():
                for key_feature, key_coffea in variables.items():
                    if (collection == "JetGoodMatched") & (key_coffea == "provenance"):
                        self.array_dict[sample][collection][key_feature] = cs[f"PartonMatched_{key_coffea}"].value
                    else:
                        self.array_dict[sample][collection][key_feature] = cs[f"{collection}_{key_coffea}"].value

                # Add padded features to the array, according to the features dictionary
                if collection in self.features_pad_dict[sample].keys():
                    for key_feature, value in self.features_pad_dict[sample][collection].items():
                        self.array_dict[sample][collection][key_feature] = value * np.ones_like(cs[f"{collection}_pt"].value)

            # The awkward arrays are zipped together to form the Momentum4D arrays.
            # If the collection is not a Momentum4D array, it is zipped as it is,
            # otherwise the Momentum4D arrays are zipped together and unflattened depending on the number of objects in the collection.
            for collection in self.array_dict[sample].keys():
                if collection in self.awkward_collections:
                    self.zipped_dict[sample][collection] = ak.unflatten(ak.zip(self.array_dict[sample][collection], with_name='Momentum4D'), cs[f"{collection}_N"].value)
                else:
                    self.zipped_dict[sample][collection] = ak.zip(self.array_dict[sample][collection], with_name='Momentum4D')
                print(f"Collection: {collection}")
                print("Fields: ", self.zipped_dict[sample][collection].fields)

            for collection in self.zipped_dict[sample].keys():
                # Pad the matched collections with None if there is no matching
                if collection in self.matched_collections_dict.keys():
                    matched_collection = self.matched_collections_dict[collection]
                    masked_arrays = ak.mask(self.zipped_dict[sample][matched_collection], self.zipped_dict[sample][matched_collection].pt==-999, None)
                    self.zipped_dict[sample][matched_collection] = masked_arrays
                    # Add the matched flag and the provenance to the matched jets
                    if collection == "JetGood":
                        is_matched = ~ak.is_none(masked_arrays, axis=1)
                        self.zipped_dict[sample][collection] = ak.with_field(self.zipped_dict[sample][collection], is_matched, "matched")
                        self.zipped_dict[sample][collection] = ak.with_field(self.zipped_dict[sample][collection], ak.fill_none(masked_arrays.prov, -1), "prov")

            # Add the remaining keys to the zipped dictionary
            if self.is_mc(sample):
                self.zipped_dict[sample]["event"] = ak.zip({"weight" : cs["weight"].value})


    @staticmethod
    def to_arrow_table(array):
        '''Convert a record array to an arrow table, with the same conversion as ak.to_parquet.'''
        import pyarrow as pa
        arrays, fields = [], []
        for name in array.fields:
            arrays.append(ak.to_arrow(array[name], list_to32=False, string_to32=True, bytestring_to32=True, allow_tensor=False))
            fields.append(pa.field(name, arrays[-1].type).with_nullable(isinstance(ak.type(array[name]).type, ak.types.OptionType)))
        return pa.Table.from_arrays(arrays, schema=pa.schema(fields))

    def save_parquet(self):
        '''Create the parquet files with the zipped dictionary.
        With the parquet schema, the ntuples are converted and written batch by batch (one row group per batch).'''

        for sample in self.samples:

            if len(self.samples) > 1:
                output_file = self.output_file.replace(".parquet", f"_{sample}.parquet")
            else:
                output_file = self.output_file

            if self.schema == "coffea":
                # The Momentum4D arrays are zipped together to form the final dictionary of arrays.
                print("Zipping the collections into a single dictionary...")
                df_out = ak.zip(self.zipped_dict[sample], depth_limit=1)
                print(f"Saving the output dataset to file: {output_file}")
                ak.to_parquet(df_out, output_file)
                continue

            import pyarrow.parquet as pq
            print(f"Saving the output dataset to file: {output_file} (batches of {self.batch_size} events)")
            writer, nwritten = None, 0
            try:
                for events in self.read_batches():
                    zipped_dict = self.build_arrays_from_ntuples(events, sample, verbose=writer is None)
                    table = self.to_arrow_table(ak.zip(zipped_dict, depth_limit=1))
                    if writer is None:
                        writer = pq.ParquetWriter(output_file, table.schema)
                    elif not table.schema.equals(writer.schema):
                        # e.g. float32 instead of float64 columns in some chunks: cast to the schema of the first batch
                        table = table.cast(writer.schema)
                    writer.write_table(table)
                    nwritten += table.num_rows
                    print(f"  {nwritten}/{self.nevents} events written", flush=True)
            except BaseException:
                # Do not leave a partial output behind
                if writer is not None:
                    writer.close()
                if os.path.exists(output_file):
                    os.remove(output_file)
                raise
            writer.close()
