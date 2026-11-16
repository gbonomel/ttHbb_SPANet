# DCTR training with two classifiers and DisCo decorrelation (script version of dctr_two_classifiers_disco.ipynb)
#
# The DCTR weight for ttbb events in the CR is the density ratio w(x) = (n_D(x) - n_B(x)) / n_T'(x),
# with D = data, B = other MC backgrounds (physical, positive weights) and T' = T * rf = ttbb scaled to the
# data - other bkg yield in the CR. It is split into two density ratios learned from samples with (mostly) positive weights:
#   - classifier data: data (1) vs T' + B (0), giving r(x) = n_D / (n_T' + n_B)
#   - classifier bkg:  B (1) vs T' (0), MC only, giving b(x) = n_B / n_T'
# and the DCTR weight is exactly w(x) = r(x) (1 + b(x)) - b(x).
#
# Decorrelation (DisCo): the data classifier is trained with
#   L = BCE + lambda * dCorr^2_w(w(x), z)|ttbb',  z = 1 (SR), 0 (CR)
# so that the DCTR weight of ttbb is independent of the region. clf_bkg is trained first and frozen.
# One model is trained per lambda, with the same clf_bkg and the same initialisation of clf_data.
import os
import json
import argparse
from copy import deepcopy
from typing import Tuple

import awkward as ak
import numpy as np
import pandas as pd
import vector
import torch
from torch import nn
from torch.optim import AdamW
import joblib
from omegaconf import OmegaConf
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ROOT)
vector.register_awkward()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="DCTR parquet input file", required=True)
parser.add_argument("-o", "--output", help="Output directory for plots and DCTR models", required=True)
parser.add_argument("-c", "--config", help="YAML file with the training settings", required=True)
parser.add_argument("--nepochs", default=None, type=int, help="Number of epochs (overrides the config)")
parser.add_argument("--load-model-folder", default=None,
                    help="Folder with the saved state_dicts to skip the training (overrides the config)")
args = parser.parse_args()

# ## Configuration
cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
if args.nepochs is not None:
    cfg["nepochs"] = args.nepochs
if args.load_model_folder is not None:
    cfg["load_model_folder"] = args.load_model_folder

dctr_input = args.input
save_folder = args.output

channel = cfg["channel"]
assert channel in ["semileptonic", "dileptonic"], f"channel must be 'semileptonic' or 'dileptonic', got '{channel}'"
cr_cuts = cfg["cr_cuts"]
nepochs = cfg["nepochs"]
patience = cfg["patience"]
batch_size = cfg["batch_size"]
lr = cfg["lr"]
weight_decay = cfg["weight_decay"]
val_fraction = cfg["val_fraction"]
seed = cfg["seed"]
hidden_data = cfg["hidden_data"]
hidden_bkg = cfg["hidden_bkg"]
lambdas_disco = cfg["lambdas_disco"]
disco_batch_size = cfg["disco_batch_size"]
n_val_disco_chunks = cfg["n_val_disco_chunks"]
lambdas_eval = cfg["lambdas_eval"] if cfg["lambdas_eval"] is not None else lambdas_disco
load_model_folder = cfg["load_model_folder"]
renormalize_closure = cfg["renormalize_closure"]
regions_by_njet = cfg["regions_by_njet"]
corner_features = cfg["corner_features"]

np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs(save_folder, exist_ok=True)
# keep a copy of the settings used for this training
OmegaConf.save(OmegaConf.create(cfg), os.path.join(save_folder, "dctr_config.yaml"))

CMAP_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]


class WeightedQuantileTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=1000, output_distribution='normal'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def save(self, filename):
        np.savez(
            filename,
            quantiles=self.quantiles_,
            reference_quantiles=self.reference_quantiles_,
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
        )

    def load(self, filename):
        data = np.load(filename)
        self.quantiles_ = data["quantiles"]
        self.reference_quantiles_ = data["reference_quantiles"]
        self.n_quantiles = int(data["n_quantiles"])
        self.output_distribution = str(data["output_distribution"])

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


def get_cr_mask(scores, cr_cuts):
    mask = (
        (scores["tthbb_vs_bkg"] >= cr_cuts["tthbb_transformed_min"]) &
        (scores["tthbb_vs_bkg"] < cr_cuts["tthbb_transformed_max"]) &
        (scores["ttlf"] < cr_cuts["ttlf_max"])
    )
    return mask

def get_sr_mask(scores, cr_cuts):
    mask = (
        (scores["tthbb_vs_bkg"] >= cr_cuts["tthbb_transformed_max"]) &
        (scores["ttlf"] < cr_cuts["ttlf_max"])
    )
    return mask


# ## Load inputs
df = ak.from_parquet(dctr_input)

weights = ak.to_numpy(df.event.weight)
njet = ak.to_numpy(ak.num(df.JetGood))

# jet multiplicity bins for the per-njet checks, starting from the dataset selection; the last bin is inclusive (>= 7)
njet_bins = list(range(int(njet.min()), 8))
def njet_label(nj):
    return f">={nj}j" if nj == njet_bins[-1] else f"{nj}j"
def njet_mask(nj_arr, nj):
    return (nj_arr >= nj) if nj == njet_bins[-1] else (nj_arr == nj)
print("njet bins:", [njet_label(nj) for nj in njet_bins])

# SPANet scores used for the CR/SR definition
spanet = df.spanet_output
pred_labels = {
    "tthbb_vs_bkg": ak.to_numpy(spanet.tthbb / (spanet.tthbb + spanet.ttbb + spanet.ttcc + spanet.ttlf)),
    "ttlf": ak.to_numpy(spanet.ttlf),
}
is_tthbb = ak.to_numpy(df.tthbb == 1)


# ## Quantile transformation of the ttHbb vs bkg score
print("Computing quantile transformation for tthbb score...")
transformer = WeightedQuantileTransformer(n_quantiles=10000, output_distribution='uniform')
transformer.fit(pred_labels["tthbb_vs_bkg"][is_tthbb], sample_weight=weights[is_tthbb])
transformer.save(os.path.join(save_folder, "tthbb_vs_bkg_quantile_transformer.npz"))
pred_labels_transformed = deepcopy(pred_labels)
pred_labels_transformed["tthbb_vs_bkg"] = transformer.transform(pred_labels["tthbb_vs_bkg"])

fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.hist(pred_labels["tthbb_vs_bkg"][is_tthbb], weights=weights[is_tthbb], bins=10, range=(0, 1), histtype="step", linewidth=3, color=CMAP_10[0], label="score", density=True)
ax.hist(pred_labels_transformed["tthbb_vs_bkg"][is_tthbb], weights=weights[is_tthbb], bins=10, range=(0, 1), histtype="step", linewidth=3, color=CMAP_10[1], label="transformed score", density=True)
ax.legend(loc='best')
ax.set_xlabel("ttHbb_vs_ttbar score")
ax.set_ylabel("Counts")
plt.savefig(os.path.join(save_folder, "ttHbb_vs_ttbar_transformed.png"), dpi=300)
plt.close(fig)


# ## Samples, physical weights and ttbb normalisation
# In the dataset the `dctr == 1` sample is data minus the other backgrounds: the other MC backgrounds are stored
# with flipped-sign weights. Here the sign is flipped back, so that the other backgrounds have their physical weights.
mask_CR = get_cr_mask(pred_labels_transformed, cr_cuts)
mask_SR = get_sr_mask(pred_labels_transformed, cr_cuts)

is_data = ak.to_numpy(df.data == 1)
is_bkg = ak.to_numpy((df.dctr == 1) & (df.data == 0))
is_ttbb = ak.to_numpy(df.dctr == 0)
assert np.all(is_data.astype(int) + is_bkg + is_ttbb == 1), "each event must be exactly one of data, other bkg, ttbb"

# physical weights: flip back the sign of the other backgrounds
weights_phys = np.where(is_bkg, -weights, weights)

# ttbb normalisation in the CR: ttbb is scaled to the data - other bkg yield, so that the DCTR learns only shape effects
W_D = weights_phys[mask_CR & is_data].sum()
W_B = weights_phys[mask_CR & is_bkg].sum()
W_T = weights_phys[mask_CR & is_ttbb].sum()
rf = (W_D - W_B) / W_T
print(f"CR yields: data = {W_D:.1f}, other bkg = {W_B:.1f}, ttbb = {W_T:.1f}")
print(f"rf = (data - other bkg) / ttbb = {rf:.4f}")
print(f"data classifier class yields: data = {W_D:.1f} vs ttbb + other bkg = {W_T * rf + W_B:.1f}")
print(f"bkg classifier class yields:  other bkg = {W_B:.1f} vs ttbb = {W_T * rf:.1f} (ratio {W_B / (W_T * rf):.3f}, learned by the classifier)")

# weights used everywhere below: physical weights, with ttbb scaled by rf (ttbb)
weights_norm = np.where(is_ttbb, weights_phys * rf, weights_phys)


# ## DCTR inputs and train/validation split
# njet is a DCTR input (the inclusive normalisation is in rf). In the semileptonic channel (>= 3 b-jets)
# the 3rd bjet pt and eta are also added; in the dileptonic channel (>= 1 b-jet) the 3rd bjet may not exist.
input_names = list(df.events.fields)
input_arrays = [ak.to_numpy(df.events[f]) for f in df.events.fields]
if channel == "semileptonic":
    input_names += ["bjet_pt_3", "bjet_eta_3"]
    input_arrays += [ak.to_numpy(df.BJetGood.pt[:, 2]), ak.to_numpy(df.BJetGood.eta[:, 2])]
input_names += ["njet"]
input_arrays += [njet]
dctr_inputs_all = np.stack(input_arrays, axis=1).astype(np.float32)  # [event, feature]
print("DCTR inputs:", input_names)

# CR arrays used for the training
idx_CR = np.flatnonzero(mask_CR)
X_cr = dctr_inputs_all[idx_CR]
data_cr, bkg_cr, ttbb_cr = is_data[idx_CR], is_bkg[idx_CR], is_ttbb[idx_CR]
w_cr = weights_norm[idx_CR]
# signed weights of the DCTR target (data - other bkg)
w_sub_cr = np.where(bkg_cr, -w_cr, w_cr)
njet_cr = njet[idx_CR]

# one train/validation split of all the events, shared by the two classifiers, the decorrelation and the closure checks
is_val_all = np.random.default_rng(seed).random(len(is_data)) < val_fraction
is_val_cr = is_val_all[idx_CR]
splits = {"train": ~is_val_cr, "val": is_val_cr}
for split, s in splits.items():
    print(f"{split:>5s}: data = {(s & data_cr).sum()}, other bkg = {(s & bkg_cr).sum()}, ttbb = {(s & ttbb_cr).sum()} events")

# input standardisation fitted on the CR training events; it is stored inside the model
scaler = StandardScaler().fit(X_cr[~is_val_cr])
joblib.dump(scaler, os.path.join(save_folder, "dctr_input_scaler.joblib"))

# ttbb events in CR and SR for the DisCo decorrelation (MC only): z = 1 for SR, 0 for CR
idx_TT = np.flatnonzero((mask_CR | mask_SR) & is_ttbb)
X_tt = dctr_inputs_all[idx_TT]
z_tt = mask_SR[idx_TT]
is_val_tt = is_val_all[idx_TT]
# DisCo weights: ttbb weights, with negative weights set to 0 (the weighted distance correlation needs non-negative weights)
# and the SR rescaled to the CR yield, so that the two regions count the same in the decorrelation
w_tt = np.clip(weights_norm[idx_TT], 0, None)
w_tt[z_tt] *= w_tt[~z_tt & ~is_val_tt].sum() / w_tt[z_tt & ~is_val_tt].sum()
print(f"ttbb for the decorrelation: CR = {(~z_tt).sum()}, SR = {z_tt.sum()} events")


def plot_inputs_closure(sel, dctr_w=None, tag="", nbins=30, out_dir=save_folder):
    # compare data - other bkg with ttbb (and ttbb x DCTR weight) for the CR events in `sel`, with ratio panels
    ncols = 3
    nrows = (len(input_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(2 * nrows, ncols, figsize=[7 * ncols, 7 * nrows],
                             gridspec_kw={"height_ratios": [3, 1] * nrows, "hspace": 0.35})
    target = sel & (data_cr | bkg_cr)
    ttbb = sel & ttbb_cr
    for i, name in enumerate(input_names):
        ax, rax = axes[2 * (i // ncols), i % ncols], axes[2 * (i // ncols) + 1, i % ncols]
        x = X_cr[:, i]
        vmin, vmax = np.nanpercentile(x[ttbb], [1, 99])
        if name == "njet":
            vmin, vmax, nb = njet_bins[0] - 0.5, njet_bins[-1] + 1.5, njet_bins[-1] - njet_bins[0] + 2
        else:
            nb = nbins
        edges = np.linspace(vmin, vmax, nb + 1)

        h_target, _ = np.histogram(x[target], bins=edges, weights=w_sub_cr[target])
        e_target = np.sqrt(np.histogram(x[target], bins=edges, weights=w_sub_cr[target]**2)[0])
        h_ttbb, _ = np.histogram(x[ttbb], bins=edges, weights=w_cr[ttbb])
        centers = 0.5 * (edges[1:] + edges[:-1])

        ax.errorbar(centers, h_target, yerr=e_target, fmt="o", color="black", markersize=4, label="data - other bkg")
        ax.stairs(h_ttbb, edges, linewidth=2, color=CMAP_10[0], label="ttbb")
        with np.errstate(divide="ignore", invalid="ignore"):
            # abs: the bin contents can be negative (signed weights, negative DCTR weights)
            rax.errorbar(centers, h_target / h_ttbb, yerr=np.abs(e_target / h_ttbb), fmt="o", color=CMAP_10[0], markersize=4)
            if dctr_w is not None:
                h_dctr, _ = np.histogram(x[ttbb], bins=edges, weights=w_cr[ttbb] * dctr_w[ttbb])
                ax.stairs(h_dctr, edges, linewidth=2, color=CMAP_10[2], label="ttbb x DCTR")
                rax.errorbar(centers, h_target / h_dctr, yerr=np.abs(e_target / h_dctr), fmt="s", color=CMAP_10[2], markersize=4)
        rax.axhline(1, color="gray", linestyle="--")
        rax.set_ylim(0.7, 1.3)
        rax.set_xlabel(name)
        rax.set_ylabel("target / ttbb", fontsize=14)
        ax.set_ylabel("Events")
        ax.legend(fontsize=12)

    for j in range(len(input_names), nrows * ncols):
        axes[2 * (j // ncols), j % ncols].set_visible(False)
        axes[2 * (j // ncols) + 1, j % ncols].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"dctr_inputs_variables_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# before training: shape differences between data - other bkg and ttbb that the DCTR has to learn
plot_inputs_closure(np.ones(len(idx_CR), dtype=bool), tag="before_dctr")


# ## DCTR model
# DCTRModel.forward(x) takes the raw inputs (in the order of `input_names`) and returns the DCTR weight.
# For a classifier with output f = sigmoid(logit), the density ratio is f / (1 - f) = exp(logit).
def make_mlp(input_dim, hidden):
    layers = []
    for h_dim in hidden:
        layers += [nn.Linear(input_dim, h_dim), nn.ReLU()]
        input_dim = h_dim
    layers.append(nn.Linear(input_dim, 1))  # logit
    return nn.Sequential(*layers)


class DCTRModel(nn.Module):
    # DCTR weight for ttbb events: w(x) = r(x) (1 + b(x)) - b(x), with
    #   r = n_data / (n_ttbb + n_bkg) from clf_data and b = n_bkg / n_ttbb from clf_bkg
    def __init__(self, input_dim, x_mean, x_scale, hidden_data, hidden_bkg):
        super().__init__()
        # input standardisation inside the model, so that the exported model takes the raw inputs
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_scale", torch.as_tensor(x_scale, dtype=torch.float32))
        # global scale of the output weight (1 unless the closure renormalisation is applied)
        self.register_buffer("w_norm", torch.tensor(1.0))
        self.clf_data = make_mlp(input_dim, hidden_data)
        self.clf_bkg = make_mlp(input_dim, hidden_bkg)

    def standardize(self, x):
        return (x - self.x_mean) / self.x_scale

    def logit_data(self, x):
        return self.clf_data(self.standardize(x)).squeeze(-1)

    def logit_bkg(self, x):
        return self.clf_bkg(self.standardize(x)).squeeze(-1)

    def components(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = torch.exp(self.logit_data(x))
        b = torch.exp(self.logit_bkg(x))
        return r, b, self.w_norm * (r * (1 + b) - b)

    def forward(self, x):
        return self.components(x)[2]


base_model = DCTRModel(len(input_names), scaler.mean_, scaler.scale_, hidden_data, hidden_bkg).to(device)
print(base_model)


# ## Training of the two sub-classifiers
# Each sub-classifier is trained with its own optimiser on its own parameters only.
# The weights are normalised globally (mean weight = 1) and the loss is the batch mean of w * BCE.
# clf_bkg is trained first and frozen. Then, for each lambda, clf_data is trained with the DisCo term computed on the
# full DCTR weight w = r (1 + b) - b of a random batch of ttbb CR + SR events at each step.
def iterate_batches(n, batch_size, shuffle):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def distance_corr(x, y, w):
    # weighted squared distance correlation dCorr^2(x, y) (Kasieczka & Shih, arXiv:2001.05310), for non-negative weights w
    w = w / w.mean()

    def centered(v):
        d = (v[:, None] - v[None, :]).abs()
        d_avg = (d * w).mean(dim=1)
        return d - d_avg[:, None] - d_avg[None, :] + (d_avg * w).mean()

    def wmean(m):
        return ((m * w).mean(dim=1) * w).mean()

    a, b = centered(x), centered(y)
    return wmean(a * b) / torch.sqrt(wmean(a * a) * wmean(b * b))


def to_tensors(X, y, w):
    return (torch.from_numpy(X).to(device),
            torch.from_numpy(y.astype(np.float32)).to(device),
            torch.from_numpy(w.astype(np.float32)).to(device))


def train_classifier(logit_fn, submodule, X, y, w, is_val, name, decor=None, lambda_decor=0.):
    # train `submodule` (the parameters used by `logit_fn`) to separate y == 1 from y == 0 with the signed weights w.
    # No class rebalancing: the classifier learns the yields ratio, exp(logit) = n_1(x) / n_0(x).
    # decor: optional decorrelation, dict(fn=model, X=..., z=..., w=..., is_val=...): loss = BCE + lambda_decor * dCorr^2(fn(X), z).
    # With lambda_decor = 0 the dCorr^2 is only monitored.
    w = w / w[~is_val].mean()
    tensors = {"train": to_tensors(X[~is_val], y[~is_val], w[~is_val]), "val": to_tensors(X[is_val], y[is_val], w[is_val])}
    print(f"[{name}] train events: {len(tensors['train'][0])}, val events: {len(tensors['val'][0])}")

    use_decor = decor is not None and lambda_decor > 0
    if decor is not None:
        dec = {split: to_tensors(decor["X"][s], decor["z"][s], decor["w"][s])
               for split, s in [("train", ~decor["is_val"]), ("val", decor["is_val"])]}
        # fixed validation subsamples, so that the validation dCorr^2 is comparable between epochs
        gen = torch.Generator().manual_seed(seed)
        val_chunks = [torch.randperm(len(dec["val"][0]), generator=gen)[:disco_batch_size].to(device) for _ in range(n_val_disco_chunks)]

    optimizer = AdamW(submodule.parameters(), lr=lr, weight_decay=weight_decay)
    log = {"train": [], "val": [], "train_dcorr": [], "val_dcorr": [], "val_total": []}
    best_val, best_epoch, best_state = np.inf, -1, None

    for epoch in range(nepochs):
        submodule.train()
        X_tr, y_tr, w_tr = tensors["train"]
        sum_wl, sum_dcorr, n_steps = 0., 0., 0
        for b in iterate_batches(len(X_tr), batch_size, shuffle=True):
            loss_ev = nn.functional.binary_cross_entropy_with_logits(logit_fn(X_tr[b]), y_tr[b], reduction="none") * w_tr[b]
            loss = loss_ev.mean()
            if use_decor:
                X_d, z_d, w_d = dec["train"]
                ib = torch.randint(len(X_d), (disco_batch_size,), device=device)
                dcorr = distance_corr(decor["fn"](X_d[ib]), z_d[ib], w_d[ib])
                loss = loss + lambda_decor * dcorr
                sum_dcorr += dcorr.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_wl += loss_ev.sum().item()
            n_steps += 1
        log["train"].append(sum_wl / w_tr.sum().item())
        log["train_dcorr"].append(sum_dcorr / n_steps if use_decor else np.nan)

        submodule.eval()
        X_va, y_va, w_va = tensors["val"]
        with torch.no_grad():
            sum_wl = sum(
                (nn.functional.binary_cross_entropy_with_logits(logit_fn(X_va[b]), y_va[b], reduction="none") * w_va[b]).sum().item()
                for b in iterate_batches(len(X_va), 8 * batch_size, shuffle=False)
            )
            if decor is not None:
                X_d, z_d, w_d = dec["val"]
                val_dcorr = np.mean([distance_corr(decor["fn"](X_d[c]), z_d[c], w_d[c]).item() for c in val_chunks])
            else:
                val_dcorr = np.nan
        log["val"].append(sum_wl / w_va.sum().item())
        log["val_dcorr"].append(val_dcorr)
        log["val_total"].append(log["val"][-1] + (lambda_decor * val_dcorr if use_decor else 0.))

        if log["val_total"][-1] < best_val:
            best_val, best_epoch, best_state = log["val_total"][-1], epoch, deepcopy(submodule.state_dict())
        if epoch % 5 == 0:
            print(f"[{name}] epoch {epoch}: train loss {log['train'][-1]:.5f}, val loss {log['val'][-1]:.5f}"
                  + (f", train dCorr^2 {log['train_dcorr'][-1]:.5f}, val dCorr^2 {val_dcorr:.5f}" if decor is not None else ""),
                  flush=True)
        if epoch - best_epoch >= patience:
            print(f"[{name}] early stopping at epoch {epoch}")
            break

    print(f"[{name}] best epoch {best_epoch}, val loss (BCE + lambda dCorr^2) {best_val:.5f}")
    submodule.load_state_dict(best_state)
    log["best_epoch"] = best_epoch

    fig, axs = plt.subplots(1, 2 if decor is not None else 1, figsize=(16 if decor is not None else 8, 6), squeeze=False)
    axs[0, 0].plot(log["train"], label="Train Loss")
    axs[0, 0].plot(log["val"], label="Val Loss")
    axs[0, 0].set_title(f"{name} BCE loss")
    if decor is not None:
        if use_decor:
            axs[0, 1].plot(log["train_dcorr"], label="Train dCorr$^2$")
        axs[0, 1].plot(log["val_dcorr"], label="Val dCorr$^2$")
        axs[0, 1].set_title(f"{name} dCorr$^2$(w, SR/CR)")
    for ax in axs[0]:
        ax.axvline(best_epoch, color="gray", linestyle="--", label="best epoch")
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"loss_{name}.png"), dpi=300)
    plt.close(fig)
    return log


# trained models per lambda; clf_bkg is trained once and shared by all of them
models = {}
if load_model_folder is not None:
    for l in lambdas_disco:
        models[l] = deepcopy(base_model)
        models[l].load_state_dict(torch.load(os.path.join(load_model_folder, f"dctr_model_lambda{l}_state_dict.pth"), map_location=device))
    print(f"Loaded models for lambdas {lambdas_disco} from {load_model_folder}")
else:
    # classifier bkg: other bkg (1) vs ttbb (0), MC only, with the physical yields ratio
    sel = bkg_cr | ttbb_cr
    log_bkg = train_classifier(base_model.logit_bkg, base_model.clf_bkg, X_cr[sel], bkg_cr[sel], w_cr[sel], is_val_cr[sel], name="clf_bkg")
    # clf_bkg is frozen: the decorrelation acts only on clf_data, through w = r (1 + b) - b
    base_model.clf_bkg.requires_grad_(False)

    # classifier data: data (1) vs ttbb + other bkg (0), equal class yields by construction of rf, with the DisCo term
    decor = {"X": X_tt, "z": z_tt, "w": w_tt, "is_val": is_val_tt}
    logs_data = {}
    for l in lambdas_disco:
        print("==========================================")
        print(f"Training clf_data with lambda = {l}")
        m = deepcopy(base_model)  # same trained clf_bkg and same initialisation of clf_data for all lambdas
        logs_data[l] = train_classifier(m.logit_data, m.clf_data, X_cr, data_cr, w_cr, is_val_cr, name=f"clf_data_lambda{l}",
                                        decor=dict(decor, fn=m), lambda_decor=l)
        models[l] = m
        torch.save(m.state_dict(), os.path.join(save_folder, f"dctr_model_lambda{l}_state_dict.pth"))


# ## Decorrelation vs lambda
# On the validation events, for each lambda:
# - decorrelation of the DCTR weight of ttbb between SR and CR: weighted mean of w in SR and CR,
#   AUC of SR vs CR using w alone (0.5 = no separation), dCorr^2
# - performance in the CR: validation BCE of clf_data (gain over ln2) and closure
@torch.no_grad()
def predict_components(model, X, batch=500000):
    model.eval()
    out = [[], [], []]
    for start in range(0, len(X), batch):
        for o, v in zip(out, model.components(torch.from_numpy(X[start:start + batch]).to(device))):
            o.append(v.cpu().numpy())
    return tuple(np.concatenate(o).astype(np.float64) for o in out)


def closure(dctr_w, sel):
    # sum(ttbb x DCTR) / sum(data - other bkg): ~1 if the DCTR weight is correctly normalised
    return np.sum(w_cr[sel & ttbb_cr] * dctr_w[sel & ttbb_cr]) / np.sum(w_sub_cr[sel & (data_cr | bkg_cr)])


def bce_data(r, sel):
    # weighted BCE of clf_data on the CR events in sel, from r = exp(logit): -log f = log(1 + 1/r), -log(1 - f) = log(1 + r)
    loss = np.where(data_cr, np.log1p(1 / r), np.log1p(r))
    return np.average(loss[sel], weights=w_cr[sel])


# fixed validation subsamples of ttbb CR + SR for dCorr^2
rng_val = np.random.default_rng(seed)
idx_val_tt = np.flatnonzero(is_val_tt)
val_chunks_tt = [rng_val.choice(idx_val_tt, disco_batch_size, replace=False) for _ in range(n_val_disco_chunks)]

summary = []
w_tt_pred = {}
for l, m in models.items():
    _, _, w_tt_pred[l] = predict_components(m, X_tt)
    r_cr_l, _, w_cr_l = predict_components(m, X_cr)
    w_pred = w_tt_pred[l]
    cr, sr = is_val_tt & ~z_tt, is_val_tt & z_tt
    with torch.no_grad():
        dcorr = np.mean([distance_corr(*(torch.from_numpy(a[c].astype(np.float32)).to(device) for a in (w_pred, z_tt, w_tt))).item()
                         for c in val_chunks_tt])
    t_val = splits["val"] & ttbb_cr
    summary.append({
        "lambda": l,
        "mean w CR": np.average(w_pred[cr], weights=w_tt[cr]),
        "mean w SR": np.average(w_pred[sr], weights=w_tt[sr]),
        "AUC SR vs CR (w)": roc_auc_score(z_tt[is_val_tt], w_pred[is_val_tt], sample_weight=w_tt[is_val_tt]),
        "dCorr^2": dcorr,
        "val BCE gain (ln2 - BCE)": np.log(2) - bce_data(r_cr_l, splits["val"]),
        "val mean w (ttbb CR)": np.average(w_cr_l[t_val], weights=w_cr[t_val]),
        "val closure": closure(w_cr_l, splits["val"]),
    })
summary = pd.DataFrame(summary).set_index("lambda")
summary.to_csv(os.path.join(save_folder, "disco_lambda_summary.csv"))
print(summary.to_string())

# DCTR weight of ttbb in CR vs SR (validation events) for each lambda
fig, axs = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)
for ax, (l, w_pred) in zip(axs[0], w_tt_pred.items()):
    for (region, sel), color in zip([("CR", is_val_tt & ~z_tt), ("SR", is_val_tt & z_tt)], [CMAP_10[0], CMAP_10[1]]):
        ax.hist(w_pred[sel], weights=w_tt[sel], bins=60, range=(0.5, 1.5), histtype="step", linewidth=2, color=color, density=True,
                label=f"ttbb {region} (mean={np.average(w_pred[sel], weights=w_tt[sel]):.3f})")
    ax.set_xlabel("DCTR weight")
    ax.set_ylabel("Normalized events (weighted)")
    ax.set_title(f"lambda = {l}")
    ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, "disco_dctr_weight_CR_SR_vs_lambda.png"), dpi=300)
plt.close(fig)

# trade-off between the CR performance and the decorrelation
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(summary["dCorr^2"], summary["val BCE gain (ln2 - BCE)"], "o-", color=CMAP_10[0])
for l, row in summary.iterrows():
    ax.annotate(f"lambda={l}", (row["dCorr^2"], row["val BCE gain (ln2 - BCE)"]), fontsize=12, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("dCorr$^2$(w, SR/CR) on ttbb (val)")
ax.set_ylabel("clf_data val BCE gain (ln2 - BCE)")
plt.savefig(os.path.join(save_folder, "disco_tradeoff.png"), dpi=300)
plt.close(fig)


# ## Evaluation and closure helpers
def plot_calibration(ax, pred, num, den, title, ylabel, nbins=20):
    # observed ratio sum(w[num]) / sum(w[den]) in bins of the predicted ratio, for train and validation.
    # num and den are signed weights per CR event (0 for events not in the class); the bins are quantiles of pred on den events
    in_den = den != 0
    edges = np.unique(np.quantile(pred[in_den], np.linspace(0, 1, nbins + 1)))
    ibin = np.clip(np.digitize(pred, edges) - 1, 0, len(edges) - 2)
    n = len(edges) - 1
    for (split, s), color in zip(splits.items(), [CMAP_10[0], CMAP_10[1]]):
        s_num, s_den = np.bincount(ibin[s], weights=num[s], minlength=n), np.bincount(ibin[s], weights=den[s], minlength=n)
        e_num, e_den = np.sqrt(np.bincount(ibin[s], weights=num[s]**2, minlength=n)), np.sqrt(np.bincount(ibin[s], weights=den[s]**2, minlength=n))
        pred_avg = np.bincount(ibin[s], weights=den[s] * pred[s], minlength=n) / s_den
        obs = s_num / s_den
        ax.errorbar(pred_avg, obs, yerr=np.abs(obs) * np.sqrt((e_num / s_num)**2 + (e_den / s_den)**2),
                    fmt="o", color=color, label=split)
    ax.plot(edges[[0, -1]], edges[[0, -1]], "k--", label="perfect calibration")
    ax.set_xlabel("predicted (bin average)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=12)


# DCTR weight regions (L/M/H): the CR is split in 3 regions by the weighted terciles of the DCTR weight of the
# ttbb events (inclusively or per jet multiplicity)
region_names = ["L", "M", "H"]
region_colors = {"L": CMAP_10[0], "M": "black", "H": CMAP_10[2]}
groups = {njet_label(nj): njet_mask(njet_cr, nj) for nj in njet_bins} if regions_by_njet else {"inclusive": np.ones(len(idx_CR), dtype=bool)}


def weighted_quantiles(values, w, q):
    order = np.argsort(values)
    cum_w = np.cumsum(w[order])
    return np.interp(q, cum_w / cum_w[-1], values[order])


def feature_edges(i, nbins=30):
    # binning of input i: integer bins for njet, otherwise the 1-99% range of the CR ttbb events
    if input_names[i] == "njet":
        return np.arange(njet_bins[0] - 0.5, njet_bins[-1] + 1.5)
    vmin, vmax = np.nanpercentile(X_cr[ttbb_cr, i], [1, 99])
    return np.linspace(vmin, vmax, nbins + 1)


def plot_features_by_region(sel, region_masks, tag, out_dir, nbins=30):
    # normalized shapes of the ttbb inputs in the L/M/H regions of the DCTR weight
    ncols = 3
    nrows = (len(input_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=[7 * ncols, 6 * nrows])
    axes = axes.flatten()
    t_all = sel & ttbb_cr
    for i, name in enumerate(input_names):
        ax = axes[i]
        edges = feature_edges(i, nbins)
        ax.hist(X_cr[t_all, i], bins=edges, weights=w_cr[t_all], density=True, histtype="step",
                linewidth=2, color="gray", linestyle="--", label="ttbb all")
        for r, m in region_masks.items():
            t = t_all & m
            ax.hist(X_cr[t, i], bins=edges, weights=w_cr[t], density=True, histtype="step",
                    linewidth=2, color=region_colors[r], label=f"ttbb {r}")
        ax.set_xlabel(name)
        ax.set_ylabel("Normalized events (weighted)")
        ax.legend(fontsize=12)
    for j in range(len(input_names), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"CR ttbb in DCTR weight regions: {tag}", fontsize="large")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"dctr_regions_features_{tag}.png"), dpi=200)
    plt.close(fig)


def plot_corner_by_region(features, sel, region_masks, tag, out_dir, nbins=20, nlevels=4):
    # correlations between the inputs in the L/M/H regions (ttbb events): 1D normalized shapes on the diagonal,
    # contours of the normalized 2D densities below the diagonal
    idx = [input_names.index(f) for f in features]
    n = len(idx)
    fig, axs = plt.subplots(n, n, figsize=(4.5 * n, 4.5 * n))
    t_all = sel & ttbb_cr
    for a, ia in enumerate(idx):
        for c, ic in enumerate(idx):
            ax = axs[a, c]
            if c > a:
                ax.set_visible(False)
                continue
            if a == c:
                for r, m in region_masks.items():
                    t = t_all & m
                    ax.hist(X_cr[t, ia], bins=feature_edges(ia, nbins), weights=w_cr[t], density=True,
                            histtype="step", linewidth=2, color=region_colors[r], label=r)
                if a == 0:
                    ax.legend(fontsize=12)
            else:
                ex, ey = feature_edges(ic, nbins), feature_edges(ia, nbins)
                xc, yc = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1])
                for r, m in region_masks.items():
                    t = t_all & m
                    h, _, _ = np.histogram2d(X_cr[t, ic], X_cr[t, ia], bins=[ex, ey], weights=w_cr[t], density=True)
                    ax.contour(xc, yc, h.T, levels=nlevels, colors=region_colors[r], linewidths=1.5)
            if a == n - 1:
                ax.set_xlabel(input_names[ic], fontsize=14)
            if c == 0 and a > 0:
                ax.set_ylabel(input_names[ia], fontsize=14)
            ax.tick_params(labelsize=10)
    fig.suptitle(f"CR ttbb in DCTR weight regions (L blue, M black, H red): {tag}", fontsize="large")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"dctr_regions_corner_{tag}.png"), dpi=150)
    plt.close(fig)


def evaluate(lambda_eval, model, out_dir):
    # evaluation plots, closure checks and TorchScript export of the model trained with lambda = lambda_eval
    os.makedirs(out_dir, exist_ok=True)
    print("==========================================")
    print(f"Evaluating the model with lambda = {lambda_eval}")

    r_all, b_all, w_dctr_all = predict_components(model, dctr_inputs_all)
    w_dctr_cr = w_dctr_all[idx_CR]
    def print_closure():
        # the weighted mean DCTR weight on ttbb should be ~1 (shape only); the yield closure of a split also contains
        # the random fluctuation of the data yield in that split, shown by the same ratio without DCTR
        for split, s in splits.items():
            t = s & ttbb_cr
            print(f"{split:>5s}: weighted mean DCTR weight (ttbb) = {np.average(w_dctr_cr[t], weights=w_cr[t]):.4f}, "
                  f"(ttbb x DCTR) / (data - other bkg) = {closure(w_dctr_cr, s):.4f}, "
                  f"without DCTR = {closure(np.ones_like(w_dctr_cr), s):.4f}")

    print_closure()

    if renormalize_closure:
        t = splits["train"] & ttbb_cr
        model.w_norm.fill_(model.w_norm.item() / np.average(w_dctr_cr[t], weights=w_cr[t]))
        r_all, b_all, w_dctr_all = predict_components(model, dctr_inputs_all)
        w_dctr_cr = w_dctr_all[idx_CR]
        print(f"renormalised the DCTR weight by w_norm = {model.w_norm.item():.4f}")
        print_closure()
        torch.save(model.state_dict(), os.path.join(save_folder, f"dctr_model_lambda{lambda_eval}_state_dict.pth"))

    # ## Export
    # done before the plots, so that a failure in the plotting does not lose the exported model.
    # TorchScript export as a single function: raw inputs [n_events, n_inputs] (order in the inputs json) to DCTR weight [n_events]
    model_cpu = deepcopy(model).cpu().eval()
    scripted = torch.jit.script(model_cpu)
    export_path = os.path.join(save_folder, f"dctr_model_lambda{lambda_eval}_scripted.pt")
    scripted.save(export_path)

    with open(os.path.join(save_folder, f"dctr_model_lambda{lambda_eval}_inputs.json"), "w") as f:
        json.dump({"input_names": input_names, "rf": float(rf), "w_norm": model_cpu.w_norm.item(), "cr_cuts": cr_cuts,
                   "lambda_disco": lambda_eval}, f, indent=2)

    # check that the exported model gives the same DCTR weight
    loaded = torch.jit.load(export_path)
    x_check = torch.from_numpy(dctr_inputs_all[:100000])
    with torch.no_grad():
        np.testing.assert_allclose(loaded(x_check).numpy(), model_cpu(x_check).numpy(), rtol=1e-5, atol=1e-6)
    print(f"Exported model saved to {export_path}")

    print("\nclosure per jet multiplicity (validation split):")
    for nj in njet_bins:
        print(f"  {njet_label(nj):>5s}: {closure(w_dctr_cr, splits['val'] & njet_mask(njet_cr, nj)):.4f}")

    for region, mask in [("CR", mask_CR), ("SR", mask_SR)]:
        t = mask & is_ttbb
        print(f"{region}: fraction of ttbb events with negative DCTR weight = {np.mean(w_dctr_all[t] < 0):.4f}, "
              f"weighted fraction = {weights_norm[t][w_dctr_all[t] < 0].sum() / weights_norm[t].sum():.4f}")

    # calibration of the two classifiers and of the DCTR weight
    r_cr, b_cr = r_all[idx_CR], b_all[idx_CR]
    zero = np.zeros_like(w_cr)
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    plot_calibration(axs[0], r_cr, np.where(data_cr, w_cr, zero), np.where(data_cr, zero, w_cr),
                     "clf_data: r = data / (ttbb + bkg)", "observed data / (ttbb + other bkg)")
    plot_calibration(axs[1], b_cr, np.where(bkg_cr, w_cr, zero), np.where(ttbb_cr, w_cr, zero),
                     "clf_bkg: b = bkg / ttbb", "observed other bkg / ttbb")
    plot_calibration(axs[2], w_dctr_cr, np.where(ttbb_cr, zero, w_sub_cr), np.where(ttbb_cr, w_cr, zero),
                     "DCTR: w = r (1 + b) - b", "observed (data - other bkg) / ttbb")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dctr_calibration.png"), dpi=300)
    plt.close(fig)

    # distributions of r, b and of the DCTR weight for ttbb events in the CR and SR
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    for ax, values, name, rng in zip(axs, [r_all, b_all, w_dctr_all], ["r (clf_data)", "b (clf_bkg)", "DCTR weight"], [(0.5, 1.5), (0, 3), (0, 2)]):
        for (region, mask), color in zip([("CR", mask_CR), ("SR", mask_SR)], [CMAP_10[0], CMAP_10[1]]):
            t = mask & is_ttbb
            ax.hist(values[t], weights=weights_norm[t], bins=60, range=rng, histtype="step", linewidth=2, color=color, density=True,
                    label=f"ttbb {region} (mean={np.average(values[t], weights=weights_norm[t]):.3f})")
        ax.set_xlabel(name)
        ax.set_ylabel("Normalized events (weighted)")
        ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dctr_weight_distributions.png"), dpi=300)
    plt.close(fig)

    # DCTR weight per jet multiplicity in the CR
    fig = plt.figure(figsize=(8, 6))
    for nj in njet_bins:
        t = mask_CR & is_ttbb & njet_mask(njet, nj)
        plt.hist(w_dctr_all[t], weights=weights_norm[t], range=(0, 2), bins=50, histtype="step", linewidth=2, density=True, label=njet_label(nj))
    plt.xlabel("DCTR weight")
    plt.ylabel("Normalized events (weighted)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "dctr_weight_bynjet.png"), dpi=300)
    plt.close(fig)

    # input variables after the DCTR reweighting: on the validation split (independent of the training) and on the full CR
    plot_inputs_closure(splits["val"], dctr_w=w_dctr_cr, tag="after_dctr_val", out_dir=out_dir)
    plot_inputs_closure(np.ones(len(idx_CR), dtype=bool), dctr_w=w_dctr_cr, tag="after_dctr_all", out_dir=out_dir)

    # ## DCTR weight regions (L/M/H)
    # region index (0, 1, 2 = L, M, H) of every CR event from its DCTR weight; the edges are computed on the ttbb events
    region_cr = np.zeros(len(idx_CR), dtype=int)
    region_edges = {}
    for label, g in groups.items():
        t = g & ttbb_cr
        region_edges[label] = weighted_quantiles(w_dctr_cr[t], w_cr[t], [1/3, 2/3])
        region_cr[g] = np.digitize(w_dctr_cr[g], region_edges[label])
        print(f"{label}: DCTR weight tercile edges = {np.round(region_edges[label], 4).tolist()}")
    region_masks = {name: region_cr == i for i, name in enumerate(region_names)}

    # yields per region: observed (data - other bkg) / ttbb compared to the mean DCTR weight of ttbb
    for split, s in splits.items():
        print(f"\n{split}:")
        for name, m in region_masks.items():
            t = s & m & ttbb_cr
            target = s & m & (data_cr | bkg_cr)
            obs = w_sub_cr[target].sum() / w_cr[t].sum()
            err = np.sqrt((w_sub_cr[target]**2).sum()) / w_cr[t].sum()
            print(f"  {name}: (data - other bkg) / ttbb = {obs:.4f} +- {err:.4f}, "
                  f"mean DCTR weight = {np.average(w_dctr_cr[t], weights=w_cr[t]):.4f}, "
                  f"after DCTR = {w_sub_cr[target].sum() / (w_cr[t] * w_dctr_cr[t]).sum():.4f}  "
                  f"(data events: {(target & data_cr).sum()}, ttbb events: {t.sum()})")

    for label, g in groups.items():
        plot_features_by_region(g, region_masks, tag=label, out_dir=out_dir)

    # data - other bkg vs ttbb (and ttbb x DCTR) in each region, full CR (the validation split has too few data events per region)
    for r, m in region_masks.items():
        plot_inputs_closure(m, dctr_w=w_dctr_cr, tag=f"region_{r}", out_dir=out_dir)

    for label, g in groups.items():
        plot_corner_by_region(corner_features, g, region_masks, tag=label, out_dir=out_dir)


for lambda_eval in lambdas_eval:
    evaluate(lambda_eval, models[lambda_eval], os.path.join(save_folder, f"eval_lambda{lambda_eval}"))
