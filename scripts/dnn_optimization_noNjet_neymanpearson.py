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

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="DCTR parquet input file", required=True)
parser.add_argument("-e", "--event", help="Event YAML file", required=True)
parser.add_argument("-o", "--output", help="Output directory for plots and DCTR model", required=True)
parser.add_argument("--nepochs", default=100, type=int, help="Number of epochs to train for")
args = parser.parse_args()

cr_cuts = {
    "tthbb_transformed_min": 0.4,
    "tthbb_transformed_max": 0.8,
    "ttlf_max": 0.2
}

# choose the list of lambdas to test for the adv training
lambdas = [1]
print("Lambdas to test for adversarial training: ", lambdas)

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

labels = {
    "tthbb": 1,
    "ttbb": 2,
    "ttcc": 3,
    "ttlf": 4,
    "ttv": 5,
    "singletop": 6,
    #"tthcc": 7,
}

# load dctr input from parquet file, and event file from yaml
df = ak.from_parquet(dctr_input)
with open(event_f, "r") as f:
    event_file = yaml.safe_load(f)

# define true and predicted (multiclassifier output) labels and weights per event
true_labels = {}
for label in labels:
    true_labels[label] = ak.to_numpy(df[f"{label}"]).astype(bool)

weights = ak.to_numpy(df.event.weight)
njet = ak.num(df.JetGood)

spanet_output_tthbb_vs_ttbb = df.spanet_output["tthbb"] / (df.spanet_output["tthbb"] + df.spanet_output["ttbb"])
spanet_output_tthbb_vs_bkg = df.spanet_output["tthbb"] / (df.spanet_output["tthbb"] + df.spanet_output["ttbb"] + df.spanet_output["ttcc"] + df.spanet_output["ttlf"])

df.spanet_output = ak.with_field(df.spanet_output, spanet_output_tthbb_vs_ttbb, "tthbb_vs_ttbb")
df.spanet_output = ak.with_field(df.spanet_output, spanet_output_tthbb_vs_bkg, "tthbb_vs_bkg")

new_labels = list(labels.keys()) 
neyman_var = ["tthbb_vs_ttbb", "tthbb_vs_bkg"]
new_labels.extend(neyman_var)

pred_labels = {}
for label in new_labels:
    pred_labels[label] = ak.to_numpy(df.spanet_output[f"{label}"])


# ======================= Quantile transformation ===========================
#
# compute on the fly the quantile transformation on tthbb score and plot the new tthbb score
print("Computing quantile transformation for tthbb score...")
is_tthbb = true_labels["tthbb"]
score_tthbb = ak.to_numpy(pred_labels["tthbb_vs_bkg"][is_tthbb])
weights_tthbb = weights[is_tthbb]
pred_labels_transformed = deepcopy(pred_labels)
transformer = WeightedQuantileTransformer(n_quantiles=10000, output_distribution='uniform')
transformer.fit(score_tthbb, sample_weight=weights_tthbb)
transformer.save(os.path.join(save_folder, "tthbb_vs_bkg_quantile_transformer.npz"))
pred_labels_transformed["tthbb_vs_bkg"] = transformer.transform(pred_labels["tthbb_vs_bkg"])

fig, ax = plt.subplots(1,1, figsize=[6,6])
ax.hist(pred_labels["tthbb_vs_bkg"][is_tthbb], weights=weights[is_tthbb], bins=10, range=(0,1), histtype="step", linewidth=3, color=CMAP_10[0], label="tthbb score", density=True)
ax.hist(pred_labels_transformed["tthbb_vs_bkg"][is_tthbb], weights=weights[is_tthbb], bins=10, range=(0,1), histtype="step", linewidth=3, color=CMAP_10[1], label="tthbb transformed score (weighted)", density=True)
ax.legend(loc='best');
ax.set_xlabel("ttHbb_vs_ttbar score")
ax.set_ylabel("Counts")
plt.savefig(os.path.join(save_folder, "ttHbb_vs_ttbar_transformed.png"), dpi=300)
plt.close()
print("Transformed tthbb score saved in", save_folder)

# now create the masks for the SR and the CR
mask_ttbb_CR = get_cr_mask(pred_labels_transformed,cr_cuts)
mask_ttbb_SR = get_sr_mask(pred_labels_transformed,cr_cuts)

# define mask for the classes for the classifier: 1 if data or other bkg, 0 if ttbb 
is_data = ak.to_numpy(ak.where(df.dctr == 1, ak.ones_like(df.dctr), ak.zeros_like(df.dctr))) == 1
is_ttbb = (df.dctr == 0)

# create list of DCTR inputs in the signal and control regions (adding 3rd bjet pt and eta)
bjet_pt_3 = df.BJetGood.pt[:,2]
bjet_eta_3 = df.BJetGood.eta[:,2]

data_cr = []
data_sr = []
for f in df.events.fields:
    data_cr.append(ak.to_numpy(df.events[f][mask_ttbb_CR]))
    data_sr.append(ak.to_numpy(df.events[f][mask_ttbb_SR]))
data_cr.append(ak.to_numpy(bjet_pt_3[mask_ttbb_CR]))
data_cr.append(ak.to_numpy(bjet_eta_3[mask_ttbb_CR]))
data_sr.append(ak.to_numpy(bjet_pt_3[mask_ttbb_SR]))
data_sr.append(ak.to_numpy(bjet_eta_3[mask_ttbb_SR]))

# use also njets as input to the DCTR but we do not reweigh based on njets then
data_cr.append(ak.to_numpy(njet[mask_ttbb_CR]))
data_sr.append(ak.to_numpy(njet[mask_ttbb_SR]))

# note that with the classifier we want to distinguish data vs ttbb in the ttbb CR
# with the adversarial network we want to distinguish ttbb in the CR vs ttbb in the CR

dctr_inputs = np.stack(data_cr).T #row per event [batch,feature]
dctr_inputs_sr_ttbb = np.stack(data_sr).T[is_ttbb[mask_ttbb_SR]]
dctr_inputs_cr_ttbb = dctr_inputs[is_ttbb[mask_ttbb_CR]]
# combine ttbb SR and CR
dctr_inputs_adv = np.concatenate([dctr_inputs_sr_ttbb,dctr_inputs_cr_ttbb ], axis=0)

# target for DCTR classifier: true if data or other bkg 0 ttbb with mask for ttbb cr
target_class = ak.to_numpy(ak.where(df.dctr == 1, ak.ones_like(df.dctr), ak.zeros_like(df.dctr)))[mask_ttbb_CR]
# target for adversarial training: 1 for ttbb in SR and 0 in CR
target_adv = np.concatenate([np.ones((dctr_inputs_sr_ttbb.shape[0])), np.zeros((dctr_inputs_cr_ttbb.shape[0]))])


# ================== inclusive reweighting ========================

# instead of reweighting based on njet, we reweight inclusively ttbb to data in one bin only

counts_ttbb, edges = np.histogram(ak.to_numpy(njet[mask_ttbb_CR])[target_class == 0], weights=weights[mask_ttbb_CR][target_class == 0], bins=[4,5,6,7,25])
counts_data, edges = np.histogram(ak.to_numpy(njet[mask_ttbb_CR])[target_class == 1], weights=weights[mask_ttbb_CR][target_class == 1], bins=[4,5,6,7,25])

print("Compute inclusive reweighting before training with binning ", edges)
reweight_factor = np.sum(counts_data) / np.sum(counts_ttbb)

training_weight = weights[mask_ttbb_CR].copy()
training_weight[target_class == 0] = training_weight[target_class == 0] * reweight_factor # removed njet reweighting

training_weight_ttbb_CR = training_weight[target_class == 0].copy()

training_weight_ttbb_SR = weights[mask_ttbb_SR & is_ttbb].copy()
training_weight_ttbb_SR =  training_weight_ttbb_SR * reweight_factor # removed njet reweighting
training_weight_ttbb_adv = np.concatenate([training_weight_ttbb_SR, training_weight_ttbb_CR])

print("Plotting before and after inclusive reweighting")
fig, ax = plt.subplots(1,1, figsize=[6,6])
ax.hist(ak.to_numpy(njet[mask_ttbb_CR])[target_class == 0], weights=weights[mask_ttbb_CR][target_class == 0], bins=[4,5,6,7,25], histtype="step", linewidth=2, color=CMAP_10[0], label="ttbb")
ax.hist(ak.to_numpy(njet[mask_ttbb_CR])[target_class == 1], weights=weights[mask_ttbb_CR][target_class == 1], bins=[4,5,6,7,25], histtype="step", linewidth=2, color=CMAP_10[1], label="data")
ax.hist(ak.to_numpy(njet[mask_ttbb_CR])[target_class == 0], weights=weights[mask_ttbb_CR][target_class == 0]*reweight_factor, bins=[4,5,6,7,25], histtype="step", linewidth=2, color=CMAP_10[2],label="ttbb reweighted")
ax.legend(loc='best');
ax.set_xlabel("nJet")
ax.set_ylabel("Counts")
plt.savefig(os.path.join(save_folder, "inclusive_reweighting_CR.png"), dpi=300)
plt.close()
print("ttbb inclusive reweighting saved in", save_folder)

# =======================================================================
#                       DCTR DNN + ADVERSARIAL NETWORK
# =======================================================================
# Standard way to normalize inputs using scikit-learn before converting them to tensors. 
# This ensures your data has a mean of 0 and a standard deviation of 1, which helps the model converge faster.

scaler = StandardScaler() # initialize and Fit Standard Scaler
# save the scaler
save_path = os.path.join(save_folder, 'dctr_input_scaler.joblib')
joblib.dump(scaler, save_path)
# fit_transform calculates mean/std AND applies the transformation
x_scaled_class = scaler.fit_transform(dctr_inputs.astype(np.float32))
# apply the same scaling used in the DCTR for the CR
x_scaled_adv = scaler.transform(dctr_inputs_adv.astype(np.float32))
print("Standard scaling of DNN inputs applied")
print(f"Original Mean: {dctr_inputs.mean():.2f}") # e.g., ~50.00
print(f"Scaled Mean:   {x_scaled_class.mean():.2f}") # e.g., ~0.00

# now convert into pytorch tensors
x_tensor = torch.from_numpy(x_scaled_class)
y_tensor = torch.from_numpy(target_class)
y_sr_tensor = torch.from_numpy(target_adv)
w_tensor = torch.from_numpy(training_weight)
w_tensor_adv = torch.from_numpy(training_weight_ttbb_adv)
# rebalance events for the SR vs CR  
w_tensor_adv[y_sr_tensor==1 ] *= (y_sr_tensor == 0).sum() / (y_sr_tensor == 1).sum()

# create dataset for DCTR classifier and adversial net
dataset_class = TensorDataset(x_tensor, y_tensor, w_tensor)
dataset_adv = TensorDataset(torch.from_numpy(x_scaled_adv), y_sr_tensor, w_tensor_adv)

# 4. Train and validation split on the dataset
train_dataset, val_dataset = torch.utils.data.random_split(dataset_class, [0.8, 0.2], )
train_dataset_adv, val_dataset_adv = torch.utils.data.random_split(dataset_adv, [0.8, 0.2], )
# 5. Create DataLoaders for train and validation
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

train_loader_adv = DataLoader(train_dataset_adv, batch_size=1024, shuffle=True)
val_loader_adv = DataLoader(val_dataset_adv, batch_size=4096, shuffle=False)

# 6. Iterate through the loader (simulating a training loop)
print("Starting iteration...")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
print(f"Train batches (adv): {len(train_loader_adv)}, Val batches (adv): {len(val_loader_adv)}")

# Simple DNN

class DCTRNet(nn.Module):
    def __init__(self, input_dim, hidden_net=[64,64,64], hidden_net_classifier=[64,64], hidden_net_adv=[64,64]):
        super(DCTRNet, self).__init__()
        layers = []
        for h_dim in hidden_net:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, input_dim))  # Final layer for binary classification
        #layers.append(nn.Sigmoid())  # Output between 0 and 1 for binary classification
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

        input_dim = self.features_dim
        adv_layers = []
        for h_dim in hidden_net_adv:
            adv_layers.append(nn.Linear(input_dim, h_dim))
            adv_layers.append(nn.ReLU())
            input_dim = h_dim
        adv_layers.append(nn.Linear(input_dim, 1))  # Final layer for binary classification
        self.model_adv = nn.Sequential(*adv_layers)
    
    def forward(self, x, return_features=False, run_adv=False):
        features = self.pre_model(x)
        if run_adv:
            adv_output = self.model_adv(features)
            if return_features:
                return adv_output, features
            return adv_output
        else:
            if return_features:
                return self.classifier(features), features
            return self.classifier(features)

def train_adversarial(model, n_epochs, train_loader, train_loader_adv, val_loader, val_loader_adv, optimizer_classifier, optimizer_adv, device, lambda_adv=1.0, wait_n_epochs_before_adv=5, save_prefix="model_adv"):
    
    log_loss = []
    log_val_loss = []
    log_adv_loss = []
    log_val_adv_loss = []
    
    # Create an iterator for the adversarial loader so we can pull batches on demand
    adv_iter = iter(train_loader_adv)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_adv_loss = 0
        
        for features, labels, weights in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            
            # --- Get a batch from the adversarial loader ---
            try:
                features_adv, labels_adv, weights_adv = next(adv_iter)
            except StopIteration:
                adv_iter = iter(train_loader_adv)
                features_adv, labels_adv, weights_adv = next(adv_iter)
            
            features_adv = features_adv.to(device)
            labels_adv = labels_adv.to(device)
            weights_adv = weights_adv.to(device)

            # ============================================================
            # STEP 1: Train Adversarial Network (Maximize discrimination)
            # ============================================================
            # Freeze the feature extractor (pre_model) and classifier
            for p in model.pre_model.parameters(): p.requires_grad = False
            for p in model.classifier.parameters(): p.requires_grad = False
            for p in model.model_adv.parameters(): p.requires_grad = True # Enable adversarial head
            
            optimizer_adv.zero_grad()
            
            # Pass adversarial data through model to get domain predictions
            # Note: The features come from pre_model (frozen here)
            outputs_adv = model(features_adv, run_adv=True).squeeze()
            
            # Loss: minimize differences between prediction and true domain label (SR vs CR)
            loss_adv = nn.functional.binary_cross_entropy_with_logits(
                outputs_adv, labels_adv.float(), reduction='none'
            )
            loss_adv = (loss_adv * weights_adv).sum() / weights_adv.sum()
            
            loss_adv.backward()
            optimizer_adv.step()
            total_adv_loss += loss_adv.item()

            # ============================================================
            # STEP 2: Train Classifier + Feature Extractor (Confuse Adversary)
            # ============================================================
            # Unfreeze features and classifier
            for p in model.pre_model.parameters(): p.requires_grad = True
            for p in model.classifier.parameters(): p.requires_grad = True
            # Freeze adversarial head (we don't want to update the discriminator here)
            for p in model.model_adv.parameters(): p.requires_grad = False 
            
            optimizer_classifier.zero_grad()
            
            # A. Main Task Loss (Classification on standard training set)
            outputs = model(features).squeeze()
            loss_cls = nn.functional.binary_cross_entropy_with_logits(
                outputs, labels.float(), reduction='none'
            )
            loss_cls = (loss_cls * weights).sum() / weights.sum()
            
            if epoch < wait_n_epochs_before_adv:
                # Just train the classifier for the first few epochs to get reasonable features before adversarial training
                loss_cls.backward()
                optimizer_classifier.step()
                total_loss += loss_cls.item()
                continue
            # B. Adversarial Loss (on the adversarial dataset)
            # We want the Feature Extractor to generate features that confuse the adversary.
            # We predict domains again using the updated features
            outputs_adv_for_confusion = model(features_adv, run_adv=True).squeeze()
            
            # To confuse: We want to maximize the entropy or maximize the error of the adversary.
            # Common trick: Flip the labels or simply subtract the loss (Gradient Reversal Layer logic).
            # Here implementing by subtracting the loss term.
            loss_adv_confusion = nn.functional.binary_cross_entropy_with_logits(
                outputs_adv_for_confusion, labels_adv.float(), reduction='none'
            )
            loss_adv_confusion = (loss_adv_confusion * weights_adv).sum() / weights_adv.sum()
            
            # Total Loss = Minimize Classification Error - Lambda * Minimize Adversarial Error
            # Maximizing Adversarial Error -> Confusing the adversary
            total_batch_loss = loss_cls - lambda_adv * loss_adv_confusion
            
            total_batch_loss.backward()
            optimizer_classifier.step()
            
            total_loss += loss_cls.item() # We usually track the main task loss
            
        # --- Logging ---
        avg_loss = total_loss / len(train_loader)
        avg_adv_loss = total_adv_loss / len(train_loader) # Normalized by main loader length
        log_loss.append(avg_loss)
        log_adv_loss.append(avg_adv_loss)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        total_val_adv_loss = 0
        
        # Lists to store adversarial validation results for ROC
        y_true_adv_list = []
        y_scores_adv_list = []
        weights_adv_list = []

        with torch.no_grad():
            # Main task validation
            for features, labels, weights in val_loader:
                features, labels, weights = features.to(device), labels.to(device), weights.to(device)
                outputs = model(features).squeeze()
                val_loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs, labels.float(), reduction='none'
                )
                val_batch_loss = (val_loss * weights).sum() / weights.sum()
                total_val_loss += val_batch_loss.item()
            
            # Adversarial validation
            for features_adv, labels_adv, weights_adv in val_loader_adv:
                features_adv, labels_adv, weights_adv = features_adv.to(device), labels_adv.to(device), weights_adv.to(device)
                outputs_adv = model(features_adv, run_adv=True).squeeze()
                val_adv_loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs_adv, labels_adv.float(), reduction='none'
                )
                val_adv_batch_loss = (val_adv_loss * weights_adv).sum() / weights_adv.sum()
                total_val_adv_loss += val_adv_batch_loss.item()
                
                # Collect for ROC (match plotting frequency)
                if epoch % 5 == 0 or epoch == n_epochs - 1:
                    y_true_adv_list.append(labels_adv.cpu().numpy())
                    y_scores_adv_list.append(torch.sigmoid(outputs_adv).cpu().numpy())
                    weights_adv_list.append(weights_adv.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_adv_loss = total_val_adv_loss / len(val_loader_adv)
        log_val_loss.append(avg_val_loss)
        log_val_adv_loss.append(avg_val_adv_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}, Adv Loss (Discrim): {avg_adv_loss:.4f}")
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, Val Adv Loss: {avg_val_adv_loss:.4f}")
        
        # save every 20 epochs and for last epoch, and plot ROC curve for adversarial validation
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            # Concatenate collected arrays
            y_true_adv = np.concatenate(y_true_adv_list)
            y_scores_adv = np.concatenate(y_scores_adv_list)
            weights_adv = np.concatenate(weights_adv_list)

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true_adv, y_scores_adv, sample_weight=weights_adv)
            
            # Sort fpr to ensure it's increasing for auc calculation if needed, 
            # though roc_curve usually returns sorted arrays. 
            # Sometimes numerical instability can cause issues.
            # Using sklearn.metrics.auc with reordering just in case.
            sort_idx = np.argsort(fpr)
            fpr_sorted = fpr[sort_idx]
            tpr_sorted = tpr[sort_idx]
            
            roc_auc = auc(fpr_sorted, tpr_sorted)

            plt.figure(figsize=(22, 6))
            plt.subplot(1, 3, 1)
            plt.plot(log_loss, label="Train Loss")
            plt.plot(log_val_loss, label="Val Loss")
            plt.title("Classifier Loss")
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(log_adv_loss, label="Train Loss")
            plt.plot(log_val_adv_loss, label="Val Loss")
            plt.title("Adversarial Loss")
            plt.legend()

            # Adversarial ROC plot
            plt.subplot(1, 3, 3)
            plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Adversarial ROC Curve')
            plt.legend(loc="lower right")
            plt.savefig(f"{save_prefix}_loss_epoch{epoch}_roc.png", dpi=300)
            plt.close()

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            torch.save(model.state_dict(), f"{save_prefix}_model_epoch{epoch}.pth")
    return log_loss, log_val_loss, log_adv_loss, log_val_adv_loss


for l_adv in lambdas:
    print("==========================================")
    print(f"\nTraining with lambda_adv = {l_adv} and n_epochs = {args.nepochs} \n")
    
    model = DCTRNet(input_dim=x_tensor.shape[1], hidden_net=[64,32], 
                                hidden_net_classifier=[32,32], hidden_net_adv=[16,16]).to(device)
    

    opt = RAdam(model.parameters(), lr=0.0001)
    opt_adv = RAdam(model.parameters(), lr=0.0001)

    log_loss, log_val_loss, log_adv_loss, log_val_adv_loss = train_adversarial(
        model, n_epochs=args.nepochs,
        train_loader=train_loader, train_loader_adv=train_loader_adv,
        val_loader=val_loader, val_loader_adv=val_loader_adv,
        optimizer_classifier=opt, optimizer_adv=opt_adv,
        device=device, lambda_adv=l_adv, wait_n_epochs_before_adv=15,
        save_prefix=f"{save_folder}/lambda{l_adv}"
    )

    print(f"\nTraining with lambda_adv = {l_adv} done \n")
    with torch.no_grad():
        y_pred_adv = torch.sigmoid(model(x_tensor.to(device))).squeeze().to('cpu')


    # ====================================================================
    #           Plotting
    #=====================================================================
    
    # plot the DCTR classifier output
    plt.hist(y_pred_adv.detach().cpu().numpy()[target_class==0], weights=training_weight[target_class==0], bins=50, histtype="step", label="ttbb")
    plt.hist(y_pred_adv.detach().cpu().numpy()[target_class==1], weights=training_weight[target_class==1], bins=50, histtype="step", label="data-other bkg")
    plt.legend()
    plt.ylabel("Events")
    plt.xlabel("classifier output")
    plt.savefig(os.path.join(save_folder, f"classifier_output_lambda{l_adv}.png"), dpi=300)
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
        
        plt.hist(dctr_w.detach().cpu().numpy()[(target_class==0) & mask_nj], weights=training_weight[(target_class==0) & mask_nj], range=(0, 2), bins=50, histtype="step", linewidth=2, label=f"{nj} jets");
    plt.xlabel("DCTR weight")
    plt.ylabel("Events")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "dctr_weight_bynjet.png"), dpi=300)
    plt.close()
    print("DCTR weight saved in ", save_folder)

    # dctr weight distribution in CR vs SR for ttbb events
    f, axs = plt.subplots(figsize=(10,8))
    axs.hist(df[mask_ttbb_CR].dctr_weight, range=(0.5, 1.5), bins=50, histtype="step", linewidth=2, color=CMAP_10[0], label="CR", density=True)
    axs.hist(df[mask_ttbb_SR].dctr_weight, range=(0.5, 1.5), bins=50, histtype="step", linewidth=2, color=CMAP_10[1], label="SR", density=True)
    axs.set_xlabel("DCTR weight")
    axs.legend()
    f.savefig(os.path.join(save_folder, "dctr_weight_CR_SR.png"),dpi=300)
    plt.close()

    # reproduce the dctr inputs histograms with reweighted ttbb
    weights = ak.to_numpy(df.event.weight)
    dctr_weight = dctr_w.detach().cpu().numpy()
    fields = list(df.events.fields) + ["bjet_pt_3", "bjet_eta_3"]
    n_fields = len(fields)
    ncols = 3
    nrows = (n_fields + ncols - 1) // ncols

    # Create full-size dctr_weight array for easier indexing
    dctr_weight_full = np.zeros(len(is_data))
    dctr_weight_full[mask_ttbb_CR] = dctr_weight

    # plot the DCTR input variables for different njet bins, comparing data vs ttbb with and without reweighting
    print("Plotting DCTR input variables for data vs ttbb in CR with and without reweighting")
        
    fig, axes = plt.subplots(nrows, ncols, figsize=[6*ncols, 5*nrows])
    axes = axes.flatten()
    
    for i, field in enumerate(fields):
        ax = axes[i]
        if field == "bjet_pt_3":
            var = bjet_pt_3
        elif field == "bjet_eta_3":
            var = bjet_eta_3
        else:
            var = df.events[field]
        
        # Compute range from data
        var_np = ak.to_numpy(var)
        vmin, vmax = np.nanpercentile(var_np, [1, 99])
        
        # Combined masks: (mask_dctr or ~mask_dctr)
        mask_dctr_nj = is_data & mask_ttbb_CR
        mask_ttbb_nj = (~is_data) & mask_ttbb_CR
        
        ax.hist(var[mask_dctr_nj], weights=weights[mask_dctr_nj], bins=20, range=(vmin, vmax), 
                histtype="step", linewidth=2, color=CMAP_10[0], label="data-other bkg")
        # ttbb without any reweighting
        ax.hist(var[mask_ttbb_nj], weights=weights[mask_ttbb_nj], bins=20, range=(vmin, vmax), 
                histtype="step", linewidth=2, color=CMAP_10[4], label="ttbb")
        # rescale only with inclusive reweighting
        ax.hist(var[mask_ttbb_nj], weights=weights[mask_ttbb_nj] * reweight_factor, bins=20, range=(vmin, vmax), 
                histtype="step", linewidth=2, color=CMAP_10[1], label="ttbb (incl reweighted)")
        # rescale ttbb with inclusive reweighting and dctr weight
        ax.hist(var[mask_ttbb_nj], 
                weights=weights[mask_ttbb_nj] * reweight_factor * dctr_weight_full[mask_ttbb_nj], 
                bins=20, range=(vmin, vmax), 
                histtype="step", linewidth=2, color=CMAP_10[2], label="ttbb (incl + dctr reweighted)")
        # only dctr reweighting
        #ax.hist(var[mask_ttbb_nj], 
        #        weights=weights[mask_ttbb_nj] * dctr_weight_full[mask_ttbb_nj], 
        #        bins=20, range=(vmin, vmax), 
        #        histtype="step", linewidth=2, color=CMAP_10[2], label="ttbb (dctr reweighted)")
        ax.set_xlabel(field)
        ax.set_ylabel("Events")
        ax.legend(fontsize=12)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        fig_name = os.path.join(save_folder, f"dctr_input_variables_lambda{l_adv}_incl_reweighted.png")
        plt.savefig(fig_name, dpi=300)
        plt.tight_layout()
        plt.close()
        print("dctr variables for ttbb CR saved in", save_folder)
    
    # plot the DCTR input variables for different njet bins, comparing ttbb in CR vs SR with njet reweighting    
    print("Plotting ttbb CR vs SR") # with njet reweighting")
        
    fig, axes = plt.subplots(nrows, ncols, figsize=[6*ncols, 5*nrows])
    axes = axes.flatten()
    
    for i, field in enumerate(fields):
        ax = axes[i]
        if field == "bjet_pt_3":
            var = bjet_pt_3
        elif field == "bjet_eta_3":
            var = bjet_eta_3
        else:
            var = df.events[field]
        
        # Compute range from data (using njet selection)
        var_np = ak.to_numpy(var)
        vmin, vmax = np.nanpercentile(var_np, [1, 99])
        
        # Combined masks: (SR or CR)
        mask_ttbb_sr = (~is_data) & mask_ttbb_SR
        mask_ttbb_cr = (~is_data) & mask_ttbb_CR
        
        # We want to normalize the CR (reweighted) histograms to match the SR yield for comparison shape
        if mask_ttbb_sr.sum() > 0:
            yield_sr = weights[mask_ttbb_sr].sum()
            
            # Normalization factor for CR
            yield_cr_rew_full = (weights[mask_ttbb_cr]).sum()
            norm_full = yield_sr / yield_cr_rew_full if yield_cr_rew_full > 0 else 1.0
        else:
            norm_njet = 1.0
            norm_full = 1.0

        # with n-jet reweighting
        # 1. Plot ttbb SR
        ax.hist(var[mask_ttbb_sr], 
                weights=weights[mask_ttbb_sr] * reweight_factor, 
                bins=20, range=(vmin, vmax), 
                histtype="step", linewidth=2, color=CMAP_10[0], label="ttbb SR (rew inclusive)")
        # 2. Plot ttbb CR
        ax.hist(var[mask_ttbb_cr], 
                weights=weights[mask_ttbb_cr] * reweight_factor * dctr_weight_full[mask_ttbb_cr] * norm_full, #* dctr_weight_full[mask_ttbb_cr] * norm_full, 
                bins=20, range=(vmin, vmax), 
                histtype="step", linewidth=2, color=CMAP_10[1], label="ttbb CR (rew inclusive)")
    
        ax.set_xlabel(field)
        ax.set_ylabel("Events")
        ax.legend(fontsize=12)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        fig_name = os.path.join(save_folder, f"dctr_input_variables_CR_SR_lambda{l_adv}_incl_reweighted.png")
        plt.savefig(fig_name, dpi=300)
        plt.tight_layout()
        plt.close()
        print("dctr variables for ttbb CR/SR saved in", save_folder)