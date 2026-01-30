#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import numpy as np
import h5py
import yaml
import awkward as ak
import numba
import pandas as pd
import vector
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import onnxruntime as ort
from multiprocessing import Pool, cpu_count

from sklearn.preprocessing import QuantileTransformer
from copy import deepcopy
import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin

vector.register_numba()
vector.register_awkward()
hep.style.use(hep.style.ROOT)

# -------------------------
# Plot settings and colors
# -------------------------
CMAP_10 = [
    "#3f90da","#ffa90e","#bd1f01","#94a4a2","#832db6",
    "#a96b59","#e76300","#b9ac70","#717581","#92dadd",
]
COLOR_ALIASES = {
    "CMS_blue": CMAP_10[0],
    "CMS_orange": CMAP_10[1],
    "CMS_red": CMAP_10[2],
    "CMS_gray": CMAP_10[3],
    "CMS_purple": CMAP_10[4],
    "CMS_brown": CMAP_10[5],
    "CMS_dark_orange": CMAP_10[6],
    "CMS_beige": CMAP_10[7],
    "CMS_dark_gray": CMAP_10[8],
    "CMS_light_blue": CMAP_10[9],
}
colors = {
    "ttHbb": COLOR_ALIASES["CMS_red"],
    "ttbb": COLOR_ALIASES["CMS_dark_orange"],
    "ttcc": COLOR_ALIASES["CMS_orange"],
    "ttlf": COLOR_ALIASES["CMS_blue"],
    "ttV": COLOR_ALIASES["CMS_brown"],
    "singleTop": COLOR_ALIASES["CMS_purple"],
}
labels = {
    "tthbb": 1,
    "ttbb": 2,
    "ttcc": 3,
    "ttlf": 4,
    "ttv": 5,
    "singletop": 6
}
nice_labels = {
    "tthbb" : "$t\\bar{t}H(b\\bar{b})$",
    "ttbb" : "$t\\bar{t}$+B",
    "ttcc" : "$t\\bar{t}$+C",
    "ttlf" : "$t\\bar{t}$+LF",
}


# -------------------------
# Helper functions
# -------------------------
def plot_score(pred_label, score, label, ax, weights=None, title=None, mask=None, log=False, ylim=None, density=False, legend=True):
    if mask is not None:
        pred_label = pred_label[mask]
        if weights is not None:
            weights = weights[mask]
    color = colors.get(label, ax._get_lines.get_next_color())
    n, bins, patches = ax.hist(pred_label, bins=50, range=(0,1),
                               weights=weights, histtype="step",
                               linewidth=2, color=color, label=label, density=density)
    if legend:
        ax.legend(fontsize=15)
    ax.set_xlabel(f"{score} SPANet score")
    ax.set_ylabel("A.U." if density else "Counts")
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    return n, bins, patches

def plot_scores(true_labels, pred_labels, plot_dir=None, weights=None, title=None, mask=None, log=False, ylim=None, density=False):
    fig, axes = plt.subplots(2, 3, figsize=[20,14], dpi=100)
    for i, score in enumerate(labels):
        ax = axes[i // 3][i % 3]
        for label in labels:
            mask_by_sample = true_labels[label]
            if mask is not None:
                mask_by_sample &= mask
            plot_score(pred_labels[score], score, label, ax=ax,
                       mask=mask_by_sample, weights=weights,
                       title=title, log=log, ylim=ylim, density=density)
    fig.tight_layout(pad=1.5)
    if plot_dir:
        filename = os.path.join(plot_dir, "classification_scores.png")
        if weights is not None:
            filename = filename.replace(".png","_weighted.png")
        if log:
            filename = filename.replace(".png","_log.png")
        if density:
            filename = filename.replace(".png","_normalised.png")
        plt.savefig(filename, dpi=300)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, plot_dir=None, norm=None, class_names=None, title=None, cmap='Blues', weights=None):
    #cm = confusion_matrix(y_true, y_pred, normalize=norm, sample_weight=weights)[1:,1:]
    #labels_ = class_names if class_names is not None else np.arange(cm.shape[0])
    label_ids   = [1, 2, 3, 4, 5, 6]
    label_names = ["ttHbb", "ttbb", "ttcc", "ttlf", "ttV", "singleTop"]
    cm = confusion_matrix(y_true, y_pred, labels=label_ids, normalize=norm, sample_weight=weights)
    plt.figure(figsize=(12,12))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=label_names, yticklabels=label_names,)
    #sns.heatmap(cm, annot=True, cmap=cmap, xticklabels=labels_, yticklabels=labels_, fmt=".2f")
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if plot_dir:
        fname = "confusion_matrix.png"
        if norm=="true":
            fname = "confusion_matrix_norm_true.png"
        elif norm=="pred":
            fname = "confusion_matrix_norm_pred.png"
        plt.savefig(os.path.join(plot_dir, fname), dpi=100)
    plt.close()

def plot_roc(true_label, pred_label, ax, label, weight=None):
    """
    Plots a ROC curve and calculates the AUC using roc_auc_score.
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(true_label, pred_label, sample_weight=weight)
    
    # Compute AUC safely
    try:
        AUC = roc_auc_score(true_label, pred_label, sample_weight=weight)
    except ValueError:
        AUC = float('nan')  # in case all labels are the same, avoid crashing

    # Plot ROC curve
    ax.plot(fpr, tpr, label=f"{label} AUC={round(AUC, 3)}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{label} ROC Curve")
    ax.legend()
    ax.grid(True)

def plot_rocs(true_labels, pred_labels, weight=None, plot_dir=None):
    """
    Plots ROC curves for all classes in the labels dictionary.
    """
    fig, axes = plt.subplots(2, 3, figsize=[18,12])
    axes = axes.flatten()
    
    for i, (score, label_index) in enumerate(labels.items()):
        ax = axes[i]
        true_label = true_labels[score]
        pred_label = pred_labels[score]
        plot_roc(true_label, pred_label, ax, score + " weighed", weight=weight)

    fig.tight_layout(pad=2.5)
    if plot_dir is not None:
        filename = os.path.join(plot_dir, "roc_curves.png")
        plt.savefig(filename, dpi=300)

def plot_correlation_matrix(numpy_array, plot_dir, feature_names=None, title="Correlation Matrix", annot=True, cmap="coolwarm"):
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(numpy_array.shape[1]-1)]
    df = pd.DataFrame(numpy_array[:,1:], columns=feature_names)
    corr_matrix = df.corr()
    plt.figure(figsize=(12,12))
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, xticklabels=feature_names, yticklabels=feature_names)
    plt.title(title)
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,"correlation_matrix.png"), dpi=100)
    plt.close()

def plot_cumulative_significance(true_scores, scores, weights, signal_index=None, bins=30, range=(0,1), xlabel="Threshold", ylabel="$S/\\sqrt{B}$", plot_dir=None):
    if signal_index is None: signal_index = 1
    thresholds = np.linspace(0,1,100)
    true_signal = (true_scores==signal_index)
    true_bkg = ~true_signal
    pred_sig_scores = scores[:,signal_index]
    pred_scores = np.argmax(scores, axis=1)
    is_signal_pred = (pred_scores==signal_index)
    score_pred_signal = scores[is_signal_pred, signal_index]
    true_scores_selected = true_scores[is_signal_pred]
    weights_selected = weights[is_signal_pred]
    cum_sig_pred = np.zeros_like(thresholds)
    cum_sig_true = np.zeros_like(thresholds)
    for i, thr in enumerate(thresholds):
        pass_signal = score_pred_signal>thr
        pass_true_signal = (pred_sig_scores>thr)&true_signal
        pass_true_bkg = (pred_sig_scores>thr)&true_bkg
        S_true = np.sum(weights[pass_true_signal])
        B_true = np.sum(weights[pass_true_bkg])
        cum_sig_true[i] = S_true/np.sqrt(B_true) if B_true>0 else 0
        S_pred = np.sum(weights_selected[pass_signal & (true_scores_selected==1)])
        B_pred = np.sum(weights_selected[pass_signal & (true_scores_selected!=1)])
        cum_sig_pred[i] = S_pred/np.sqrt(B_pred) if B_pred>0 else 0
    fig, ax = plt.subplots(figsize=[10,8])
    ax.plot(thresholds, cum_sig_pred, marker='o', color=CMAP_10[0], linestyle='-', label="Predicted as ttHbb")
    ax.plot(thresholds, cum_sig_true, marker='o', color=CMAP_10[1], linestyle='-', label="ttHbb node")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Cumulative Significance")
    ax.legend(fontsize=15)
    if plot_dir:
        plt.savefig(os.path.join(plot_dir,"cumulative_significance.png"), dpi=300)
    plt.close()

def compute_s_over_sqrtb(y_true, scores, weights, signal_index=None, bins=30, range=(0,1)):
    if signal_index is None: signal_index=1
    if isinstance(bins,int):
        bins = np.linspace(range[0],range[1],bins+1)
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    s_over_sqrtb = np.zeros(len(bin_centers))
    for i,(low,high) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin = (scores>=low)&(scores<high)
        S = np.sum(weights[in_bin & (y_true==signal_index)])
        B = np.sum(weights[in_bin & (y_true!=signal_index)])
        s_over_sqrtb[i] = S/np.sqrt(B) if B>0 else 0
    return s_over_sqrtb

# function to prepare the spanet input and event weights from parquet and event file
def load_inputs_from_parquet(parquet_file, event_file, labels):
    with open(event_file) as f:
        event_file = yaml.safe_load(f)
    df = ak.from_parquet(parquet_file)
    print(f"Loaded {parquet_file}.")
    df = df[:10000]  # for testing, limit to first 10k events
    print(f"Number of events in parquet: {len(df)}")
    # load the variable names from event file
    jet_inputs    = event_file["INPUTS"]["SEQUENTIAL"]["Jet"]
    met_inputs    = event_file["INPUTS"]["GLOBAL"]["Met"]
    lepton_inputs = event_file["INPUTS"]["GLOBAL"]["Lepton"]
    event_inputs  = event_file["INPUTS"]["GLOBAL"]["Event"]

    # load inputs from parquet and stack in awkward for ort input
    jet_pad_for_mask = ak.pad_none(df.JetGood["pt"],16,axis=1,clip=True)[:,:,None] 
    jet_pad = ak.concatenate([ak.pad_none(df.JetGood[var],16,axis=1,clip=True)[:,:,None] for var in jet_inputs], axis=-1)
    jet_input = ak.fill_none(jet_pad, 0)
    lepton_input = ak.concatenate([df.LeptonGood[var][:, None] for var in lepton_inputs], axis=-1)[:, None, :]
    met_input = ak.concatenate([df.PuppiMET[var][:, None] for var in met_inputs], axis=-1)[:, None, :]
    event_input = ak.concatenate([df.events[var][:, None] for var in event_inputs], axis=-1)[:, None, :]

    # convert to format that spanet accepts
    jet_data = ak.to_numpy(jet_input).astype("float32")
    met_data = ak.to_numpy(met_input).astype("float32")
    lepton_data = ak.to_numpy(lepton_input).astype("float32")
    event_data = ak.to_numpy(event_input).astype("float32")
    
    # create the masks: jets paddded to 16
    jet_pad_for_mask = ak.pad_none(df.JetGood["pt"],16,axis=1,clip=True)
    jet_mask = ~ak.is_none(jet_pad_for_mask,axis=1).to_numpy()
    mask_global = np.ones((met_data.shape[0], 1), dtype=bool)
    lepton_mask  = mask_global
    met_mask  = mask_global
    event_mask = mask_global
    
    # log normalize the variables according to the event file 
    jet_var_normalize = [var for var in jet_inputs if event_file["INPUTS"]["SEQUENTIAL"]["Jet"][var] == "log_normalize"]
    met_var_normalize = [var for var in met_inputs if event_file["INPUTS"]["GLOBAL"]["Met"][var] == "log_normalize"]
    lepton_var_normalize = [var for var in lepton_inputs if event_file["INPUTS"]["GLOBAL"]["Lepton"][var] == "log_normalize"]
    event_var_normalize = [var for var in event_inputs if event_file["INPUTS"]["GLOBAL"]["Event"][var] == "log_normalize"]

    for i, var in enumerate(jet_inputs):
        if var in jet_var_normalize:
            jet_data[:, :, i] = np.log1p(jet_data[:, :, i])
    for i, var in enumerate(met_inputs):
        if var in met_var_normalize:
            met_data[:, :, i] = np.log1p(met_data[:, :, i])
    for i, var in enumerate(lepton_inputs):
        if var in lepton_var_normalize:
            lepton_data[:, :, i] = np.log1p(lepton_data[:, :, i])
    for i, var in enumerate(event_inputs):
        if var in event_var_normalize:
            event_data[:, :, i] = np.log1p(event_data[:, :, i])

    weights = ak.to_numpy(df.event.weight)
    true_labels = { }
    for label in labels.keys():
        true_labels[label] = ak.to_numpy(df[f"{label}"]).astype(bool)

    spanet_inputs = {
        "Jet_data": jet_data,
        "Jet_mask": ak.to_numpy(jet_mask),
        "Met_data": met_data,
        "Met_mask": ak.to_numpy(met_mask),
        "Lepton_data": lepton_data,
        "Lepton_mask": ak.to_numpy(lepton_mask),
        "Event_data": event_data,
        "Event_mask": ak.to_numpy(event_mask)
    }
    print("Prepared SPANet inputs from parquet.")
    return spanet_inputs, weights, true_labels

# class for weighted quantile transformer 
class WeightedQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=1000, output_distribution='normal'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(np.array([self.quantiles_, self.reference_quantiles_]), f)

    def load(self, filename):
        extension = os.path.splitext(filename)[1]
        if not extension == '.pkl':
            raise ValueError(f"Invalid file extension '{os.path.splitext(filename)[1]}'. Only '.pkl' files are supported.")
        self.quantiles_, self.reference_quantiles_ = np.load(filename, allow_pickle=True)
    
    def _weighted_quantiles(self, X, weights):
        # Calculate weighted quantiles
        sorted_indices = np.argsort(X)
        X_sorted = X[sorted_indices]
        weights_sorted = weights[sorted_indices]
        cum_weights = np.cumsum(weights_sorted) / np.sum(weights_sorted)
        
        # Interpolate to get quantiles
        quantiles = np.interp(np.linspace(0, 1, self.n_quantiles), cum_weights, X_sorted)
        return quantiles
    
    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is None:
            raise ValueError("Sample weights must be provided.")
        
        self.quantiles_ = self._weighted_quantiles(X, sample_weight)
        
        if self.output_distribution == 'normal':
            self.reference_quantiles_ = norm.ppf(np.linspace(0, 1, self.n_quantiles))
        elif self.output_distribution == 'uniform':
            self.reference_quantiles_ = np.linspace(0, 1, self.n_quantiles)
        else:
            raise ValueError(f"Unknown output distribution '{self.output_distribution}'.")
        
        return self
    
    def transform(self, X):
        # Interpolate based on weighted quantiles
        transformed_X = np.interp(X, self.quantiles_, self.reference_quantiles_)
        return transformed_X

# function to run quantile transformer on tthbb score and save output in pkl file
def run_weighted_quantile_transformer(tthbb_score, weights, output_folder, n_quantiles=10000, output_distribution='uniform'):
    transformer = WeightedQuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)
    transformer.fit(tthbb_score, sample_weight=weights)
    filename = os.path.join(output_folder, "tthbb_quantile_transformer.pkl")
    transformer.save(filename)

# -------------------------
# ONNX inference worker
# -------------------------
def run_chunk(spanet_inputs, spanet_outputs, onnx_file, batch_size, sess_opts_dict=None, providers=None):
    """
    Run ONNX inference on a chunk of data.
    
    Args:
        spanet_inputs: Dictionary with keys from load_inputs_from_parquet:
                       Jet_data, Jet_mask, Met_data, Met_mask, 
                       Lepton_data, Lepton_mask, Event_data, Event_mask
        spanet_outputs: List of output names to request from the model
        onnx_file: Path to the ONNX model file
        batch_size: Batch size for inference
        sess_opts_dict: Dictionary with 'intra' and 'inter' thread counts
        providers: List of ONNX runtime providers
    
    Returns:
        numpy array with inference results
    """
    if sess_opts_dict is None:
        sess_opts_dict = {"intra": 1, "inter": 1}
    if providers is None:
        providers = ["CPUExecutionProvider"]
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = sess_opts_dict["intra"]
    sess_options.inter_op_num_threads = sess_opts_dict["inter"]
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_file, sess_options=sess_options, providers=providers)

    # Extract data and masks from spanet_inputs dictionary
    jet_data = spanet_inputs["Jet_data"]
    jet_mask = spanet_inputs["Jet_mask"]
    met_data = spanet_inputs["Met_data"]
    met_mask = spanet_inputs["Met_mask"]
    lepton_data = spanet_inputs["Lepton_data"]
    lepton_mask = spanet_inputs["Lepton_mask"]
    event_data = spanet_inputs["Event_data"]
    event_mask = spanet_inputs["Event_mask"]
    
    n_events = jet_data.shape[0]
    out = np.empty((n_events, 7), dtype=np.float32)
    
    for i in range(0, n_events, batch_size):
        j = min(i + batch_size, n_events)
        out[i:j] = session.run(
            spanet_outputs,
            {
                "Jet_data": jet_data[i:j],
                "Jet_mask": jet_mask[i:j],
                "Met_data": met_data[i:j],
                "Met_mask": met_mask[i:j],
                "Lepton_data": lepton_data[i:j],
                "Lepton_mask": lepton_mask[i:j],
                "Event_data": event_data[i:j],
                "Event_mask": event_mask[i:j],
            }
        )[0]
    
    return out

# -------------------------
# Main function
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="""
    Parallel ONNX inference for SPANet-style ttH analyses.

    IMPORTANT:
    • Use --workers > 1 ONLY for CPU inference
    • For GPU inference, always use --workers 1 and increase --batch-size
    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
    Examples
    --------
    CPU inference on 16 cores:
    python run_inference.py \\
        --p input.parquet \\
        --event-file event.yaml \\
        --onnx model.onnx \\
        --workers 16

    GPU inference:
    python run_inference.py \\
        --p input.parquet \\
        --event-file event.yaml \\
        --onnx model.onnx \\
        --batch-size 8192 \\
        --workers 1
    """,
    )
    parser.add_argument("--p", required=True, help="Input parquet file")
    parser.add_argument("--event-file", required=True, help="YAML configuration file")
    parser.add_argument("--onnx", required=True, help="ONNX model file")
    parser.add_argument("--out-folder", required=True, help="Output folder to save scores and figures")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for inference")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel CPU workers")
    parser.add_argument("--gpu", action="store_true", help="Force GPU only")
    parser.add_argument("--qt", action="store_true", help="Apply quantile transformer to tthbb score and save to pkl file")
    parser.add_argument("--save-scores", action="store_true", help="Save output scores to .parquet file")
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    # Load inputs from parquet file
    spanet_inputs, weights, true_labels = load_inputs_from_parquet(args.p, args.event_file, labels)
    Nevents = spanet_inputs["Jet_data"].shape[0]

    # -------------------------
    # Inference
    # -------------------------
    print(f"Running inference on {Nevents} events...")
    print("It may take a while depending on the number of events and batch size.")
    if args.gpu:
        print("Using GPU for inference.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        print(f"Using {args.workers} CPU workers for inference.")
        providers = ["CPUExecutionProvider"]
    sess_opts = {"intra": 1, "inter": 1}
    spanet_outputs = ["EVENT/signal"]
    
    # Run inference
    spanet_out = run_chunk(
        spanet_inputs=spanet_inputs,
        spanet_outputs=spanet_outputs,
        onnx_file=args.onnx,
        batch_size=args.batch_size,
        sess_opts_dict=sess_opts,
        providers=providers
    )
    print("Inference completed.")
    
    # Save output scores
    if args.save_scores:
        data = pd.DataFrame({
            "tthbb":      spanet_out[:, 1],
            "ttbb":       spanet_out[:, 2],
            "ttcc":       spanet_out[:, 3],
            "ttlf":       spanet_out[:, 4],
            "ttv":        spanet_out[:, 5],
            "singletop":  spanet_out[:, 6],
            "weight":     weights,
            #"true_label":       y_true, # not saved, parquet file needs to be loaded after in the dctr 
        })

        out_path = os.path.join(args.out_folder, "spanet_onnx_inference.parquet")
        data.to_parquet(out_path, index=False)
        print(f"Scores and weights saved to {out_path}.")
    # -------------------------
    # Compute labels
    # -------------------------
    y_pred = np.argmax(spanet_out,axis=1)
    pred_labels = {}
    for label, index in labels.items():
        pred_labels[label] = spanet_out[:,index]
    is_tthbb = true_labels["tthbb"]

    if args.qt:
        print("Running weighted quantile transformer on tthbb score...")
        run_weighted_quantile_transformer(
            tthbb_score=pred_labels["tthbb"][is_tthbb],
            weights=weights,
            output_folder=args.out_folder,
            n_quantiles=10000,
            output_distribution='uniform'
        )
        print(f"Quantile transformer saved to {args.out_folder}/tthbb_quantile_transformer.pkl")
    # -------------------------
    # Plots
    # -------------------------
    #print("Generating evaluation plots...")
    #print(" - Classification scores")
    #plot_scores(true_labels, pred_labels, plot_dir=args.out_folder, weights=weights, density=True)
    #plot_scores(true_labels, pred_labels, plot_dir=args.out_folder, weights=weights, log=True)
    #print(" - ROC curves")
    #plot_rocs(true_labels, pred_labels, weight=weights, plot_dir=args.out_folder)
    #print(" - Confusion matrices")
    #plot_confusion_matrix(true_labels, y_pred, plot_dir=args.out_folder, norm="true", class_names=labels, weights=weights)
    #plot_confusion_matrix(true_labels, y_pred, plot_dir=args.out_folder, norm="pred", class_names=labels, weights=weights)
    #print(" - Correlation matrices and cumulative significance")
    #for label, class_index in labels.items():
    #    mask = true_labels==class_index
    #    plot_correlation_matrix(spanet_out[mask], plot_dir=args.out_folder, feature_names=list(labels.keys()), title=f"{label} Correlation Matrix")
    #plot_cumulative_significance(true_labels, spanet_out, weights, plot_dir=args.out_folder)
    #print(f"All plots saved in {args.out_folder}.")

if __name__=="__main__":
    main()
