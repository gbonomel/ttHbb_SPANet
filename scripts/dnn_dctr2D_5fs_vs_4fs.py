import awkward as ak
import numba
import numpy as np
import pandas as pd
import awkward as ak
import vector
import numba
import torch
from torch import nn 
from torch.optim import RAdam
import joblib
import yaml
import argparse

from sklearn.metrics import roc_curve, auc
vector.register_numba()
vector.register_awkward()
import os

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ROOT)

from importlib import reload
from sklearn.preprocessing import QuantileTransformer
from copy import deepcopy
import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

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
        (scores["tthbb"] >= cr_cuts["tthbb_transformed_min"]) &
        (scores["tthbb"] < cr_cuts["tthbb_transformed_max"]) &
        (scores["ttlf"] < cr_cuts["ttlf_max"])
    )
    return mask 

def get_sr_mask(scores, cr_cuts):
    mask = (
        (scores["tthbb"] >= cr_cuts["tthbb_transformed_max"]) &
        (scores["ttlf"] < cr_cuts["ttlf_max"])
    )
    return mask

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="DCTR parquet input file", required=True)
parser.add_argument("-e", "--event", help="Event YAML file", required=True)
parser.add_argument("-o", "--output", help="Output directory for plots and DCTR model", required=True)
parser.add_argument("--nepochs", default=100, type=int, help="Number of epochs to train for")
args = parser.parse_args()

cr_cuts = {
    "tthbb_transformed_min": 0.5,
    "tthbb_transformed_max": 0.8,
    "ttlf_max": 0.2
}

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dctr_input = args.input
event_f = args.event
save_folder = args.output

# ensure save folder exists
os.makedirs(save_folder, exist_ok=True)

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

spanet_labels = {
    "tthbb": 1,
    "ttbb": 2,
    "ttcc": 3,
    "ttlf": 4,
}

true_samples = ['tthbb', 'tthnonbb', '5fs', '4fs', 'ttcc', 'ttlf', 'ttv', 'singletop', 'vv', 'vjets']

# load dctr input from parquet file, and event file from yaml
df = ak.from_parquet(dctr_input)

# define true and predicted (multiclassifier output) labels and weights per event
true_labels = {}
pred_labels = {}

weights = ak.to_numpy(df.event.weight)
njet = ak.num(df.JetGood)
for label in spanet_labels.keys():
    pred_labels[label] = ak.to_numpy(df.spanet_output[f"{label}"])
for label in true_samples:
    true_labels[label] = ak.to_numpy(df[f"{label}"]).astype(bool)
# ======================= Quantile transformation ===========================
#
# compute on the fly the quantile transformation on tthbb score and plot the new tthbb score
print("Computing quantile transformation for tthbb score...")
is_tthbb = true_labels["tthbb"]
score_tthbb = ak.to_numpy(pred_labels["tthbb"][is_tthbb])
weights_tthbb = weights[is_tthbb]
pred_labels_transformed = deepcopy(pred_labels)
transformer = WeightedQuantileTransformer(n_quantiles=10000, output_distribution='uniform')
transformer.fit(score_tthbb, sample_weight=weights_tthbb)
transformer.save(os.path.join(save_folder, "tthbb_quantile_transformer.npz"))
pred_labels_transformed["tthbb"] = transformer.transform(pred_labels["tthbb"])

fig, ax = plt.subplots(1,1, figsize=[6,6])
ax.hist(pred_labels["tthbb"][is_tthbb], weights=weights[is_tthbb], bins=10, range=(0,1), histtype="step", linewidth=3, color=CMAP_10[0], label="tthbb score", density=True)
ax.hist(pred_labels_transformed["tthbb"][is_tthbb], weights=weights[is_tthbb], bins=10, range=(0,1), histtype="step", linewidth=3, color=CMAP_10[1], label="tthbb transformed score (weighted)", density=True)
ax.legend(loc='best');
ax.set_xlabel("ttHbb score")
ax.set_ylabel("Counts")
plt.savefig(os.path.join(save_folder, "ttHbb_transformed.png"), dpi=300)
plt.close()
print("Transformed tthbb score saved in", save_folder)

# now create the masks for the SR and the CR
mask_ttbb_CR = get_cr_mask(pred_labels_transformed,cr_cuts)
mask_ttbb_SR = get_sr_mask(pred_labels_transformed,cr_cuts)

# create list of DCTR inputs in the signal and control regions (adding 3rd bjet pt and eta)
bjet_pt_3 = df.BJetGood.pt[:,2]
bjet_eta_3 = df.BJetGood.eta[:,2]

data_cr = []
for f in df.events.fields:
    data_cr.append(ak.to_numpy(df.events[f][mask_ttbb_CR]))
data_cr.append(ak.to_numpy(bjet_pt_3[mask_ttbb_CR]))
data_cr.append(ak.to_numpy(bjet_eta_3[mask_ttbb_CR]))
data_cr.append(ak.to_numpy(njet[mask_ttbb_CR]))

# note that with the classifier we want to distinguish data vs ttbb in the ttbb CR
# with the adversarial network we want to distinguish ttbb in the CR vs ttbb in the CR

dctr_inputs_cr = np.stack(data_cr).T #row per event [batch,feature]

# classes for classifier: true (1) if ttbb 5FS, 0 if ttbb 4FS
is_ttbb_5FS = (df["5vs4fs"] == 1)
is_ttbb_4FS = (df["5vs4fs"] == 0)

dctr_inputs_cr_4fs = dctr_inputs_cr[is_ttbb_4FS[mask_ttbb_CR]]
dctr_inputs_cr_5fs = dctr_inputs_cr[is_ttbb_5FS[mask_ttbb_CR]]

# target for DCTR classifier: true if 5FS (1), false (0) 4FS ttbb in cr
x_dnn = np.concatenate([dctr_inputs_cr_4fs, dctr_inputs_cr_5fs], axis=0)
target_dnn = np.concatenate([np.zeros(len(dctr_inputs_cr_4fs)), np.ones(len(dctr_inputs_cr_5fs))])

# ================== reweighting ========================

# instead of reweighting based on njet, we reweight inclusively ttbb to data in one bin only

counts_5fs = np.sum(target_dnn == 1)
counts_4fs = np.sum(target_dnn == 0)

weights_cr_4fs = ak.to_numpy(weights[mask_ttbb_CR][is_ttbb_4FS[mask_ttbb_CR]])
weights_cr_5fs = ak.to_numpy(weights[mask_ttbb_CR][is_ttbb_5FS[mask_ttbb_CR]])

weighted_nevents_4fs = ak.sum(weights[mask_ttbb_CR][is_ttbb_4FS[mask_ttbb_CR]])
weighted_nevents_5fs = ak.sum(weights[mask_ttbb_CR][is_ttbb_5FS[mask_ttbb_CR]])

reweight_nevents = weighted_nevents_4fs / weighted_nevents_5fs

fig, ax = plt.subplots(figsize=(8, 7))
ax.hist(njet[mask_ttbb_CR][is_ttbb_4FS[mask_ttbb_CR]], weights=weights_cr_4fs, bins=[4,5,6,7,25], histtype="step", linewidth=2, label="ttbb 4FS")
ax.hist(njet[mask_ttbb_CR][is_ttbb_5FS[mask_ttbb_CR]], weights=weights_cr_5fs, bins=[4,5,6,7,25], histtype="step", linewidth=2, label="ttbb 5FS")
ax.hist(njet[mask_ttbb_CR][is_ttbb_5FS[mask_ttbb_CR]], weights=weights_cr_5fs*reweight_nevents, bins=[4,5,6,7,25], histtype="step", linewidth=2, label="ttbb 5FS (reweighted njet, \nnormalized to 4FS)")
ax.set_xlabel("nJets")
ax.set_ylabel("Events")
ax.legend(loc='best', fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(save_folder, "njet_5vs4FS_reweight.png"), dpi=300)
plt.close()

training_weight = np.concatenate([
    weights_cr_4fs,
    weights_cr_5fs * reweight_nevents]) # * reweight_5fs


# =======================================================================
#                       DCTR SIMPLE DNN 5FS (1) vs 4FS (0) CLASSIFIER
# =======================================================================

x_numpy = x_dnn.astype(np.float32) 
scaler = StandardScaler() 
save_path = os.path.join(save_folder, 'dctr_input_scaler.joblib')
joblib.dump(scaler, save_path)
x_scaled_class = scaler.fit_transform(x_numpy.astype(np.float32))
print("Standard scaling of DNN inputs applied")
print(f"Original Mean: {x_numpy.mean():.2f}") # e.g., ~50.00
print(f"Scaled Mean:   {x_scaled_class.mean():.2f}") # e.g., ~0.00

# now convert into pytorch tensors
x_tensor = torch.from_numpy(x_scaled_class)
y_tensor = torch.from_numpy(target_dnn)
w_tensor = torch.from_numpy(training_weight)
dataset_class = TensorDataset(x_tensor, y_tensor, w_tensor)

# 4. Train and validation split on the dataset
train_dataset, val_dataset = torch.utils.data.random_split(dataset_class, [0.8, 0.2], )
# 5. Create DataLoaders for train and validation
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

# 6. Iterate through the loader (simulating a training loop)
print("Starting iteration...")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Simple DNN

from torch import nn 

class DCTRNet(nn.Module):
    def __init__(self, input_dim, hidden_net=[64,64,64], hidden_net_classifier=[64,64]):
        super(DCTRNet, self).__init__()
        layers = []
        for h_dim in hidden_net:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, input_dim))  # Final layer for binary classification
        self.pre_model = nn.Sequential(*layers)
        self.features_dim = hidden_net[-1]

        input_dim = self.features_dim
        layers_classifier = []
        for h_dim in hidden_net_classifier:
            layers_classifier.append(nn.Linear(input_dim, h_dim))
            layers_classifier.append(nn.ReLU())
            input_dim = h_dim
        layers_classifier.append(nn.Linear(input_dim, 1))  # Final layer for binary classification
        self.classifier = nn.Sequential(*layers_classifier)
    
    def forward(self, x, return_features=False):
        features = self.pre_model(x)
        if return_features:
            return self.classifier(features), features
        return self.classifier(features)

def train(model, n_epochs,  train_loader, val_loader, optimizer, device, save_prefix=None):
    model.train()
    
    log_loss = []
    log_val_loss = []
    for epoch in range(n_epochs):
        total_loss = 0
        for features, labels, weights in train_loader:
            features, labels, weights = features.to(device), labels.to(device), weights.to(device)
            
            optimizer.zero_grad()  # Clear gradients
            outputs = model(features).squeeze()  # Forward pass
            loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels.float(), reduction='none')  # Compute loss
            #print(loss.shape)
            #print(weights.shape)
            loss = (loss * weights).sum()/weights.sum()  # Apply weights and average
            #print(loss, loss.shape)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            #print(f"Batch Loss: {loss.item():.4f}")
            total_loss += loss.item()  # Accumulate weighted loss
        
        avg_loss = total_loss / len(train_loader)  # Average loss over dataset
        log_loss.append(avg_loss)
        print(f"{epoch} Training Loss: {avg_loss:.4f}")

        # validation 
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for features, labels, weights in val_loader:
                features, labels, weights = features.to(device), labels.to(device), weights.to(device)
                outputs = model(features).squeeze()
                val_loss = nn.functional.binary_cross_entropy_with_logits(outputs, labels.float(), reduction='none')
                val_loss = (val_loss * weights).sum()/weights.sum()
                total_val_loss += val_loss.item() 
        log_val_loss.append(total_val_loss / len(val_loader))
        print(f"{epoch} Validation Loss: {total_val_loss / len(val_loader):.4f}")
        # plot every 20 epochs
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            plt.figure()
            plt.plot(log_loss, label="Train Loss")
            plt.plot(log_val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.savefig(f"{save_prefix}_loss_epoch{epoch}_roc.png", dpi=300)
            plt.close()
        # save model every 20 epochs
        if epoch % 20== 0 or epoch == n_epochs - 1:
            torch.save(model.state_dict(), f"{save_prefix}_model_epoch{epoch}.pth")

    return log_loss, log_val_loss

model = DCTRNet(input_dim=x_tensor.shape[1], hidden_net=[32,32],  hidden_net_classifier=[32]).to(device)
opt = RAdam(model.parameters(), lr=0.0001)

log_loss, log_val_loss = train(
    model, n_epochs=args.nepochs,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=opt,
    device=device,
    save_prefix=f"{save_folder}/5vs4fs"
)
with torch.no_grad():
    y_pred_adv = torch.sigmoid(model(x_tensor.to(device))).squeeze().to('cpu')


# ====================================================================
#           Plotting
#=====================================================================

# plot the DCTR classifier output
plt.hist(y_pred_adv.detach().cpu().numpy()[target_dnn==0], weights=training_weight[target_dnn==0], bins=50, histtype="step", label="ttbb 4FS")
plt.hist(y_pred_adv.detach().cpu().numpy()[target_dnn==1], weights=training_weight[target_dnn==1], bins=50, histtype="step", label="ttbb 5FS")
plt.legend()
plt.ylabel("Events")
plt.xlabel("classifier output")
plt.savefig(os.path.join(save_folder, f"classifier_output_5vs4fs.png"), dpi=300)
plt.close()
print("DCTR classifier output saved in ", save_folder)

dctr_w  = y_pred_adv / (1-y_pred_adv)

f, (ax, bx) = plt.subplots(1, 2, figsize=(16,8))
ax.hist(df.dctr_weight, bins=100, range=(0,3))
bx.hist(df.dctr_weight, bins=100,range=(0,3))
ax.set_yscale("log")
ax.set_xlabel("DCTR weight")
bx.set_xlabel("DCTR weight")
f.savefig(os.path.join(save_folder, "dctr_weight.png"),dpi=300)
plt.close()

# plot dctr weight for different njets
plt.figure()
for nj in range(4,8):
    if nj ==7:
        mask_nj = ak.to_numpy(njet[mask_ttbb_CR] >= nj)
    else:
        mask_nj = ak.to_numpy(njet[mask_ttbb_CR] == nj)
    
    plt.hist(dctr_w.detach().cpu().numpy()[(target_dnn==0) & mask_nj], weights=training_weight[(target_dnn==0) & mask_nj], range=(0, 2), bins=50, histtype="step", linewidth=2, label=f"{nj} jets");
plt.xlabel("5FS vs 4FS weight")
plt.ylabel("Events")
plt.legend()
plt.savefig(os.path.join(save_folder, "dctr_weight_5vs4fs.png"), dpi=300)
plt.close()
print("DCTR weight saved in ", save_folder)

# dctr weight distribution in CR vs SR for ttbb events
f, axs = plt.subplots(figsize=(10,8))
axs.hist(df[mask_ttbb_CR].dctr_weight, range=(0.5, 1.5), bins=50, histtype="step", linewidth=2, color=CMAP_10[0], label="CR", density=True)
axs.hist(df[mask_ttbb_SR].dctr_weight, range=(0.5, 1.5), bins=50, histtype="step", linewidth=2, color=CMAP_10[1], label="SR", density=True)
axs.set_xlabel("5FS vs 4FS weight")
axs.legend()
f.savefig(os.path.join(save_folder, "dctr_weight_CR_SR.png"),dpi=300)
plt.close()