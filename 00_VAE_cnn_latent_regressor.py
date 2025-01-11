#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  10 23:07:02 2025

@author: marcin
"""
import torch
import torch as pt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

def apply_transformation(x, on_off_transformation):
    tmp = x
    # tmp = tmp**0.25
    # tmp =  np.log10(X_[row])
    if(on_off_transformation == 1):
        tmp = (tmp - np.mean(tmp)) / np.std(tmp)
    else:
        tmp = x
    return tmp
 
def build_latent_regressor(in_dim, hidden_dims, out_dim, dropout_prob, regressor_architecture_string):
    if regressor_architecture_string == "MLP":
        layers = []
        current_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    elif regressor_architecture_string == "LSTM":
        class StackedLSTM(nn.Module):
            def __init__(self, in_dim, hidden_dims, out_dim, dropout_prob):
                super(StackedLSTM, self).__init__()
                self.lstm_layers = nn.ModuleList()
                input_dim = in_dim
                for hidden_dim in hidden_dims:
                    self.lstm_layers.append(nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True))
                    input_dim = hidden_dim
                
                self.dropout = nn.Dropout(dropout_prob)
                self.fc = nn.Linear(hidden_dims[-1], out_dim)
        
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
        
                for lstm_layer in self.lstm_layers:
                    x, _ = lstm_layer(x)
        
                last_hidden_state = x[:, -1, :]
                last_hidden_state = self.dropout(last_hidden_state)
                out = self.fc(last_hidden_state)
                return out
            
        return StackedLSTM(in_dim, hidden_dims, out_dim, dropout_prob)
    
    elif regressor_architecture_string == "CNN":
        class StackedCNN(nn.Module):
            def __init__(self, in_dim, hidden_dims, out_dim, dropout_prob, kernel_size=3, stride=1, padding=1):
                super(StackedCNN, self).__init__()
                self.conv_layers = nn.ModuleList()
                input_channels = 1
                for hidden_dim in hidden_dims:
                    self.conv_layers.append(nn.Conv1d(input_channels, hidden_dim, kernel_size, stride, padding))
                    self.conv_layers.append(nn.BatchNorm1d(hidden_dim))
                    self.conv_layers.append(nn.ReLU())
                    self.conv_layers.append(nn.Dropout(dropout_prob))
                    input_channels = hidden_dim
                
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(hidden_dims[-1], out_dim)

            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                
                for layer in self.conv_layers:
                    x = layer(x)
                
                x = self.global_pool(x)
                x = x.squeeze(-1)
                out = self.fc(x)
                return out
        
        return StackedCNN(in_dim, hidden_dims, out_dim, dropout_prob)
    
    elif regressor_architecture_string == "Transformer":
        class StackedTransformer(nn.Module):
            def __init__(self, in_dim, hidden_dims, out_dim, dropout_prob, num_heads=4):
                super(StackedTransformer, self).__init__()
                self.embedding = nn.Linear(in_dim, hidden_dims[0])  # Initial embedding to first layer's hidden dim
                self.transformer_layers = nn.ModuleList()
                
                for i in range(len(hidden_dims) - 1):
                    self.transformer_layers.append(
                        nn.TransformerEncoderLayer(
                            d_model=hidden_dims[i],
                            nhead=num_heads,
                            dim_feedforward=hidden_dims[i + 1],
                            dropout=dropout_prob,
                        )
                    )
                self.fc = nn.Linear(hidden_dims[-1], out_dim)

            def forward(self, x):
                if x.dim() == 2:  # (batch_size, feature_dim)
                    x = x.unsqueeze(1)  # Add seq_length dimension
                
                x = self.embedding(x)  # Project input to hidden_dims[0]
                for layer in self.transformer_layers:
                    x = layer(x)  # Pass through each transformer layer
                
                x = x.mean(dim=1)  # Global average pooling across sequence
                out = self.fc(x)  # Regression output
                return out
        
        return StackedTransformer(in_dim, hidden_dims, out_dim, dropout_prob)

    else:
        raise ValueError(f"Unsupported architecture_string: {regressor_architecture_string}")

        
class ConvVAERegressor(nn.Module):
    def __init__(
        self,
        input_channels          : int,
        input_length            : int,
        kernel_size             : int,
        latent_dim              : int,
        encoder_channels        : list,
        decoder_channels        : list,
        regressor_hidden_dims   : list,  
        dropout_prob            : float,
        regressor_architecture_string : str
    ):
        """
        input_channels: e.g. 1 (for a single-channel 1D signal)
        input_length:   length of the 1D signal
        kernel_size:    kernel size
        latent_dim:     size of the latent space (z-dimension)
        encoder_channels: list of out_channels for the encoder conv blocks
                          e.g., [16, 32, 64]
        decoder_channels: list of out_channels for the decoder convT blocks
                          e.g., [64, 32, 16] (symmetry) or something else
        regressor_hidden_dims: list of hidden layer sizes for the regressor MLP
                                e.g. [128, 64] => (latent_dim -> 128 -> 64 -> 1)
        dropout_prob:   dropout probability to help with regularization
        """
        super().__init__()

        # -------------------------------------------------
        # 1) Encoder
        # -------------------------------------------------
        self.encoder_layers = nn.ModuleList()
        current_channels = input_channels

        # Build encoder: Conv1d -> BatchNorm -> ReLU -> Dropout
        for out_channels in encoder_channels:
            self.encoder_layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1
                )
            )
            self.encoder_layers.append(nn.BatchNorm1d(out_channels))
            self.encoder_layers.append(nn.ReLU())
            self.encoder_layers.append(nn.Dropout(dropout_prob))
            current_channels = out_channels

        # Dynamically figure out the flattened size after encoding
        dummy_input = pt.zeros(1, input_channels, input_length)
        with pt.no_grad():
            x = dummy_input
            for layer in self.encoder_layers:
                x = layer(x)
            self.flatten_channels = x.size(1)
            self.flatten_length = x.size(2)
            self.flatten_size = self.flatten_channels * self.flatten_length

        # Latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # -------------------------------------------------
        # 2) Decoder
        # -------------------------------------------------
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        self.decoder_layers = nn.ModuleList()
        prev_channels = self.flatten_channels

        # Build decoder: ConvTranspose1d -> BatchNorm -> ReLU -> Dropout
        for out_channels in decoder_channels:
            self.decoder_layers.append(
                nn.ConvTranspose1d(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1
                )
            )
            self.decoder_layers.append(nn.BatchNorm1d(out_channels))
            self.decoder_layers.append(nn.ReLU())
            self.decoder_layers.append(nn.Dropout(dropout_prob))
            prev_channels = out_channels

        # Final layer to reconstruct X
        self.final_conv = nn.Conv1d(
            in_channels=decoder_channels[-1],
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # -------------------------------------------------
        # 3) Regression head (MLP)
        # -------------------------------------------------
        # Dynamically build the regressor MLP (latent_dim -> ... -> 1)
        # using the `built_latent_regressor` helper function.

        
        self.regressor_net = build_latent_regressor(
            in_dim = latent_dim, 
            hidden_dims = regressor_hidden_dims,
            out_dim = 1, 
            dropout_prob = dropout_prob,
            regressor_architecture_string = regressor_architecture_string)

    def reparameterize(self, mu, logvar):
        std = pt.exp(0.5 * logvar)
        eps = pt.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # ---------------- Encoder ----------------
        for layer in self.encoder_layers:
            x = layer(x)

        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to (B, flatten_size)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # ---------------- Decoder ----------------
        x_decoded = self.fc_decode(z)  # shape: (B, flatten_size)
        x_decoded = x_decoded.view(batch_size, self.flatten_channels, self.flatten_length)

        for layer in self.decoder_layers:
            x_decoded = layer(x_decoded)

        x_pred = self.final_conv(x_decoded)  # (B, input_channels, input_length)

        # ---------------- Regressor (MLP) ----------------
        y_pred = self.regressor_net(z).squeeze(-1)  # shape: (B,)

        return x_pred, y_pred, mu, logvar        


 

def vae_regression_loss(x_true, x_pred, y_true, y_pred, mu, logvar, alpha = 1, betha=1.0, gamma =1.0, epsilon=1e-6):
    """
    Args:
        x_true:  (B, C, L) - Ground truth for reconstruction
        x_pred:  (B, C, L) - Predicted reconstruction
        y_true:  (B,) or (B, 1) - Ground truth for regression
        y_pred:  (B,) - Predicted regression output
        mu:      (B, latent_dim) - Mean from VAE encoder
        logvar:  (B, latent_dim) - Log variance from VAE encoder

        alpha:   float - Weight for regression loss
        betha:   float - Weight for reconstruction loss
        gamma:   float - Weight for KL term
        epsilon: float - Small value to avoid numerical instability
    Returns:
        total_loss: Weighted total loss
        recon_loss: Reconstruction loss
        reg_loss:   Regression loss
        kl_loss:    KL divergence loss
    """
    # 1) Reconstruction Loss (MSE on X)
    recon_loss = F.mse_loss(x_pred, x_true, reduction='mean')

    # 2) Regression Loss (MSE on y)
    reg_loss = F.mse_loss(y_pred, y_true, reduction='mean')

    # 3) KL Divergence
    kl_loss = -0.5 * pt.sum(1 + logvar - mu.pow(2) - logvar.exp() + epsilon, dim=1)  # Add epsilon for stability
    kl_loss = pt.mean(kl_loss, dim=0)  # Average over the batch

    # Weighted sum
    total_loss = alpha*reg_loss + betha*recon_loss + gamma*kl_loss

    return total_loss, recon_loss, reg_loss, kl_loss


def load_and_process_data(file_path, delta_min, delta_max, n_samples, transformation, n_elements):
    data = pd.read_pickle(file_path)
    cond_delta = (data['delta'] >= delta_min) & (data['delta'] <= delta_max)
    data = data[cond_delta]
    data = data.sample(n_samples)
    data.reset_index(drop=False, inplace=True)

    X_ = data["trajectory"]
    y_ = data["delta"]

    X_processed, y_processed = [], []
    for row in range(X_.shape[0]):
        if np.any(~np.isnan(X_[row])):
            transformed = apply_transformation(X_[row], transformation)
            X_processed.append(transformed)
            y_processed.append(y_[row])

    X_processed = np.array(X_processed)[:, -n_elements:]
    y_processed = np.array(y_processed)

    # Convert to PyTorch tensors and add channel dimension
    X_tensor = pt.tensor(X_processed, dtype=pt.float32).unsqueeze(1)  # [B, C, L]
    y_tensor = pt.tensor(y_processed, dtype=pt.float32)

    return X_tensor, y_tensor

 
#%%
# Configurations
on_off_transformation = 1
N_samples = {
    'train': 20000,
    'val': 2000,
    'test': 2000
}

N_elements = 80
delta_bounds = {
    'train': (0, 10),
    'val': (0, 10),
    'test': (0, 10)
}
data_paths = {
    'train': "./data_pandas/data_trajectories_train.pkl",
    'val': "./data_pandas/data_trajectories_val.pkl",
    'test': "./data_pandas/data_trajectories_test.pkl"
}



# Load and process datasets
X_train, y_train = load_and_process_data(
    data_paths['train'], delta_bounds['train'][0], delta_bounds['train'][1], N_samples['train'], on_off_transformation, N_elements
)

X_val, y_val = load_and_process_data(
    data_paths['val'], delta_bounds['val'][0], delta_bounds['val'][1], N_samples['val'], on_off_transformation, N_elements
)

X_test, y_test = load_and_process_data(
    data_paths['test'], delta_bounds['test'][0], delta_bounds['test'][1], N_samples['test'], on_off_transformation, N_elements
)

 

##############################################################
# Hyperparameters
learning_rate = 1e-4
beta = 1e-3            # KL divergence weight
alpha =  1e-1         # regression loss weight
epochs = 100
batch_size = 1024
 


# Hyperparameters model
input_channels = X_train.shape[1]
input_length = X_train.shape[2]

latent_dim = X_train.shape[-1]
encoder_channels = [128, 64, 32, 32]
decoder_channels = encoder_channels[::-1]
regressor_hidden_dims = [128, 64, 128]

dropout = 0.1

# beta = 1e-3            # KL divergence weight
# alpha =  1e-1         # regression loss weight

set_0 = {
            # Architecture
            "encoder_channels"              :   [128, 64, 32, 32],
            "decoder_channels"              :   [32, 32, 64, 128],
            "kernel_size"                   :   3,
            "latent_regressor_hidden_dims"  :   [128, 64, 128],
            "latent_dim"                    :   X_train.shape[2]*4,
            "latent_regressor_architecture" :   "MLP",
 
            # Optimizer
            "dropout"                       :   0.1,
            "learning_rate"                 :   1e-4,
            "batch_size"                    :   256,
            "epochs"                        :   100,
            
            # Loss function
            "alpha"                         :   1e-1,       # regression loss weight
            "betha"                         :   1,    # reconstruction loss weight
            "gamma"                         :   1e-3,    # KL divergence weight
        }


set_1 = {
            # Architecture
            "encoder_channels"              :   [64, 64, 32, 32],
            "decoder_channels"              :   [32, 32, 64, 128],
            "kernel_size"                   :   3,
            "latent_regressor_hidden_dims"  :   [128, 64, 128],
            "latent_dim"                    :   X_train.shape[2]*4,
            "latent_regressor_architecture" :   "CNN",
 
            # Optimizer
            "dropout"                       :   0.1,
            "learning_rate"                 :   1e-4,
            "batch_size"                    :   256,
            "epochs"                        :   100,
            
            # Loss function
            "alpha"                         :   1e-1,   # regression loss weight
            "betha"                         :   1,      # reconstruction loss weight
            "gamma"                         :   1e-3    # KL divergence weight
        }


set_1 = {
            # Architecture
            "encoder_channels"              :   [64, 64, 32, 32],
            "decoder_channels"              :   [32, 32, 64, 128],
            "kernel_size"                   :   3,
            "latent_regressor_hidden_dims"  :   [128, 64, 128],
            "latent_dim"                    :   X_train.shape[2]*4,
            "latent_regressor_architecture" :   "LSTM",
 
            # Optimizer
            "dropout"                       :   0.1,
            "learning_rate"                 :   1e-4,
            "batch_size"                    :   256,
            "epochs"                        :   100,
            
            # Loss function
            "alpha"                         :   1e-1,   # regression loss weight
            "betha"                         :   1,      # reconstruction loss weight
            "gamma"                         :   1e-3    # KL divergence weight
        }
 
 
set_2 = {
            # Architecture
            "encoder_channels"              :   [64, 64, 32, 32],
            "decoder_channels"              :   [32, 32, 64, 128],
            "kernel_size"                   :   3,
            "latent_regressor_hidden_dims"  :   [128, 64, 128],
            "latent_dim"                    :   X_train.shape[2]*4,
            "latent_regressor_architecture" :   "CNN",
 
            # Optimizer
            "dropout"                       :   0.1,
            "learning_rate"                 :   1e-4,
            "batch_size"                    :   256,
            "epochs"                        :   100,
            
            # Loss function
            "alpha"                         :   1e-1,   # regression loss weight
            "betha"                         :   1,      # reconstruction loss weight
            "gamma"                         :   1e-3    # KL divergence weight
        }


set_3 = {
            # Architecture
            "encoder_channels"              :   [64, 64, 32, 32],
            "decoder_channels"              :   [32, 32, 64, 128],
            "kernel_size"                   :   3,
            "latent_regressor_hidden_dims"  :   [128, 64, 128],
            "latent_dim"                    :   X_train.shape[2]*4,
            "latent_regressor_architecture" :   "MLP",
 
            # Optimizer
            "dropout"                       :   0.1,
            "learning_rate"                 :   1e-4,
            "batch_size"                    :   256,
            "epochs"                        :   100,
            
            # Loss function
            "alpha"                         :   1e-1,   # regression loss weight
            "betha"                         :   1,      # reconstruction loss weight
            "gamma"                         :   1e-3    # KL divergence weight
        }



configurations = {
                    "set_0"     : set_0,
                    "set_1"     : set_1,
                    "set_2"     : set_2,
                    "set_3"     : set_3,
                }
    
for configuration_index, (configuration_key, configuration) in enumerate(configurations.items()):

    encoder_channels = configuration["encoder_channels"]
    decoder_channels = configuration["decoder_channels"]
    kernel_size      = configuration["kernel_size"]
    latent_regressor_hidden_dims  = configuration["latent_regressor_hidden_dims"]
    latent_dim       = configuration["latent_dim"]
    latent_regressor_architecture = configuration["latent_regressor_architecture"]
 
    dropout          = configuration["dropout"]
    learning_rate    = configuration["learning_rate"]

    alpha            = configuration["alpha"]
    betha            = configuration["betha"]
    gamma            = configuration["gamma"]
    batch_size       = configuration["batch_size"]
    epochs           = configuration["epochs"]

 

    ##############################################################
    
    
    # Initialize model
    model = ConvVAERegressor(
        input_channels=X_train.shape[1],
        input_length=X_train.shape[2],
        kernel_size = kernel_size,
        latent_dim=latent_dim,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        regressor_hidden_dims=latent_regressor_hidden_dims,  # for example
        dropout_prob=dropout,
        regressor_architecture_string = latent_regressor_architecture
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
     
    # For logging
    history_train_total = []
    history_train_recon = []
    history_train_reg = []
    history_train_kl = []
    
    history_val_total = []
    history_val_recon = []
    history_val_reg = []
    history_val_kl = []
    
    # Training loop
    for epoch in range(epochs):
        # ---------------------------
        # TRAINING
        # ---------------------------
        model.train()
    
        total_epoch_loss = 0
        recon_epoch_loss = 0
        reg_epoch_loss   = 0
        kl_epoch_loss    = 0
    
        # Train set: Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i : i + batch_size]  # shape (B, C, L)
            batch_y = y_train[i : i + batch_size]  # shape (B,)
    
            optimizer.zero_grad()
            x_pred, y_pred, mu, logvar = model(batch_X)
    
            total_loss, recon_loss, reg_loss, kl_loss = vae_regression_loss(
                batch_X, 
                x_pred, 
                batch_y, 
                y_pred, 
                mu, 
                logvar,
                alpha=alpha,
                betha=betha, 
                gamma=gamma
            )
    
            total_loss.backward()
            optimizer.step()
    
            total_epoch_loss += total_loss.item()
            recon_epoch_loss += recon_loss.item()
            reg_epoch_loss   += reg_loss.item()
            kl_epoch_loss    += kl_loss.item()
    
        # Compute average losses for the epoch (training)
        n_batches = len(X_train) / batch_size
        history_train_total.append(total_epoch_loss / n_batches)
        history_train_recon.append(recon_epoch_loss / n_batches)
        history_train_reg.append(reg_epoch_loss / n_batches)
        history_train_kl.append(kl_epoch_loss / n_batches)
    
        # Print training info
        print(configuration_key, latent_regressor_architecture)
        print(f"Epoch {epoch+1:03d} | "
              f"Train -> Total: {history_train_total[-1]:.4f} | "
              f"Recon: {history_train_recon[-1]:.4f} | "
              f"Reg: {history_train_reg[-1]:.4f} | "
              f"KL: {history_train_kl[-1]:.4f}")
    
        # ---------------------------
        # VALIDATION
        # ---------------------------
        model.eval()
    
        val_total_epoch_loss = 0
        val_recon_epoch_loss = 0
        val_reg_epoch_loss   = 0
        val_kl_epoch_loss    = 0
    
        with pt.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X_val = X_val[i : i + batch_size]
                batch_y_val = y_val[i : i + batch_size]
    
                x_pred_val, y_pred_val, mu_val, logvar_val = model(batch_X_val)
    
                total_loss_val, recon_loss_val, reg_loss_val, kl_loss_val = vae_regression_loss(
                    batch_X_val, 
                    x_pred_val, 
                    batch_y_val, 
                    y_pred_val, 
                    mu_val, 
                    logvar_val,
                    alpha=alpha,
                    betha=betha, 
                    gamma=gamma
                )
    
                val_total_epoch_loss += total_loss_val.item()
                val_recon_epoch_loss += recon_loss_val.item()
                val_reg_epoch_loss   += reg_loss_val.item()
                val_kl_epoch_loss    += kl_loss_val.item()
    
        # Compute average losses for the epoch (validation)
        n_val_batches = len(X_val) / batch_size
        history_val_total.append(val_total_epoch_loss / n_val_batches)
        history_val_recon.append(val_recon_epoch_loss / n_val_batches)
        history_val_reg.append(val_reg_epoch_loss / n_val_batches)
        history_val_kl.append(val_kl_epoch_loss / n_val_batches)
    
        # Print validation info
        print(f"          | "
              f"Val   -> Total: {history_val_total[-1]:.4f} | "
              f"Recon: {history_val_recon[-1]:.4f} | "
              f"Reg: {history_val_reg[-1]:.4f} | "
              f"KL: {history_val_kl[-1]:.4f}")
        
    print("Done")
    
     
     
    # Visualize results
    model.eval()
    df_reconstruction_on_train = []
    with pt.no_grad():
        for i in range(len(X_train)):
            X_row = X_train[i:i+1]
            y_true = y_train[i:i+1]
            x_pred, y_pred, loc, logscale = model(X_row)
            df_reconstruction_on_train.append({
                "loss"            : pt.mean((X_row - x_pred) ** 2).item(),
                "y_true"          : y_true.item(),
                "y_pred"          : y_pred.item(),
                "X_true"          : X_row.squeeze().cpu().numpy(),
                "X_reconstructed" : x_pred.squeeze().cpu().numpy(),
                "abs_y_true_y_pred_square"     : np.abs(y_pred.item() - y_true.item())**2
    
                
            })
    
    df_reconstruction_on_train = pd.DataFrame(df_reconstruction_on_train)
    
     
     
    # Visualize results
    model.eval()
    df_reconstruction_on_test = []
    with pt.no_grad():
        for i in range(len(X_test)):
            X_row = X_test[i:i+1]
            y_true = y_test[i:i+1]
            x_pred, y_pred, loc, logscale = model(X_row)
            
            idx_permuted1 = np.random.permutation(X_row.shape[2])               
            X_row_permuted  = X_row[:,:,idx_permuted1]
            x_pred_permuted, y_pred_permuted, loc, logscale = model(X_row_permuted)
                    
            df_reconstruction_on_test.append({
                "loss"                      : pt.mean((X_row - x_pred) ** 2).item(),
                "y_true"                    : y_true.item(),
                "y_pred"                    : y_pred.item(),
                "y_pred_permuted"           : y_pred_permuted.item(),
                
                "X_true"                    : X_row.squeeze().cpu().numpy(),
                "X_reconstructed"           : x_pred.squeeze().cpu().numpy(),
    
                "X_true_permuted"           : X_row_permuted.squeeze().cpu().numpy(),
                "X_reconstructed_permuted"  : x_pred_permuted.squeeze().cpu().numpy(),
    
                
                "abs_y_true_y_pred_square"          : np.abs(y_pred.item() - y_true.item())**2,
                "abs_y_true_y_pred_permuted_square" : np.abs(y_pred_permuted.item() - y_true.item())**2,
                
            })
            print(y_pred, y_pred_permuted)
    
    df_reconstruction_on_test = pd.DataFrame(df_reconstruction_on_test)
     
    
    
    #%%
    # Calculate averages
     
    results_test = (
        df_reconstruction_on_test
        .groupby('y_true')['y_pred']
        .agg(['mean', 'std'])   # You can add more aggregations here if you want
        .reset_index()
    )
        
    
     
    results_test_MSE = (
        df_reconstruction_on_test
        .groupby('y_true')['abs_y_true_y_pred_square']
        .agg(['mean', 'std'])   # You can add more aggregations here if you want
        .reset_index()
    )
    
    
    results_test_permuted_MSE = (
        df_reconstruction_on_test
        .groupby('y_true')['abs_y_true_y_pred_permuted_square']
        .agg(['mean', 'std'])   # You can add more aggregations here if you want
        .reset_index()
    )
     
     
    
    results_test_permuted = (
        df_reconstruction_on_test
        .groupby('y_true')['y_pred_permuted']
        .agg(['mean', 'std'])   # You can add more aggregations here if you want
        .reset_index()
    )
    
     
     
    
    results_train = (
        df_reconstruction_on_train
        .groupby('y_true')['y_pred']
        .agg(['mean', 'std'])   # You can add more aggregations here if you want
        .reset_index() 
    )
    
    results_train_MSE = (
        df_reconstruction_on_train
        .groupby('y_true')['abs_y_true_y_pred_square']
        .agg(['mean', 'std'])   # You can add more aggregations here if you want
        .reset_index() 
    )
    
    
    results_train = results_train.dropna(how='any')
    results_test = results_test.dropna(how='any')
    results_test_permuted = results_test.dropna(how='any')
    
    results_train_MSE = results_train.dropna(how='any')
    results_test_MSE = results_test.dropna(how='any')
    results_test_permuted = results_test.dropna(how='any')
    
 
    # Plot validation reconstructions
    fig, ax = plt.subplots(4, 3, figsize=(18, 14))
    fig_title_string = "MLP on latent space | Transformation = {:d}".format(on_off_transformation) 
    fit_title_string = fig_title_string +   " | parameters set : " + str(configuration_key)
    fig_title_string = "alpha = {:2.4f} | betha = {:2.4f} | gamma = {:2.4f}".format(alpha, betha, gamma)
    fig_title_string = fig_title_string + " | latent regressor :" + latent_regressor_architecture
    fig.suptitle(fig_title_string)
    samples_train = np.random.choice(range(0,df_reconstruction_on_train.shape[0]), 1)
    samples_test = np.random.choice(range(0, df_reconstruction_on_test.shape[0]), 1)
    gap = 10
    for shift, idx in enumerate(samples_train):
        print(idx)
        x_true = df_reconstruction_on_train['X_true'][idx] + shift * gap
        x_reconstructed = df_reconstruction_on_train['X_reconstructed'][idx] + shift * gap
        ax[0,0].plot(x_true, label='True')
        ax[0,0].plot(x_reconstructed, linestyle='--', label='Reconstructed')
        title_string = "Train reconstruction" 
        ax[0,0].set_title(title_string)
    
    for shift, idx in enumerate(samples_test):
        x_true = df_reconstruction_on_test['X_true'][idx] + shift * gap
        x_reconstructed = df_reconstruction_on_test['X_reconstructed'][idx] + shift * gap
        ax[0,1].plot(x_true, label='True')
        ax[0,1].plot(x_reconstructed, linestyle='--', label='Reconstructed')
        title_string = "Test reconstruction"
        ax[0,1].set_title(title_string)
        
        
    for shift, idx in enumerate(samples_test):
        x_true = df_reconstruction_on_test['X_true'][idx] + shift * gap
        x_reconstructed = df_reconstruction_on_test['X_reconstructed'][idx] + shift * gap
        
        x_true_permuted = df_reconstruction_on_test['X_true_permuted'][idx] + shift * gap
        x_reconstructed_permuted = df_reconstruction_on_test['X_reconstructed_permuted'][idx] + shift * gap
        
        ax[0,2].plot(x_true, ':', label='True')
        ax[0,2].plot(x_true_permuted,  label='True')
        
        # ax[0,2].plot(x_reconstructed, linestyle='--', label='Reconstructed')
        ax[0,2].plot(x_reconstructed_permuted, linestyle='--', label='Reconstructed')
        title_string = "Test permuted reconstruction"
        ax[0,2].set_title(title_string)    
 
    ax[1,0].plot(results_train['y_true'], results_train['mean'], 'o', label = 'train')
    ax[1,1].plot(results_test['y_true'], results_test['mean'], 'x', label = 'test')
    ax[1,2].plot(results_test_permuted['y_true'], results_test_permuted['mean'], 'x', label = 'test')
    
    
    ax[2,0].plot(results_train['y_true'], np.sqrt(results_train_MSE['mean']), 'o', label = 'sqrt [mean (y_true-y_pred)**2]')
    ax[2,1].plot(results_test['y_true'], np.sqrt(results_test_MSE['mean']), 'x', label = 'sqrt [mean (y_true-y_pred)**2]')
    ax[2,2].plot(results_test_permuted['y_true'], np.sqrt(results_test_permuted_MSE['mean']), 'x', label = 'sqrt [mean (y_true-y_pred_premtued)**2]')
    
    ax[2,0].legend()
    ax[2,1].legend()
    ax[2,2].legend()
    
    # y_max = np.max(np.square(results_test_permuted_MSE['mean']))
    # y_min = 0
    # ax[2,2].set_ylim([0, y_max])
    # ax[2,1].set_ylim([0, y_max])
    # ax[2,0].set_ylim([0, y_max])
    
    ax[1,0].plot(results_test['y_true'],results_test['y_true'])
    ax[1,1].plot(results_test['y_true'],results_test['y_true'])
    ax[1,2].plot(results_test['y_true'],results_test['y_true'])
    
    ax[1,0].fill_between(
        results_train['y_true'],
        results_train['mean'] - results_train['std'],
        results_train['mean'] + results_train['std'],
        alpha=0.2,  # adjust opacity as desired
        color='blue'
    )
    
    ax[1,1].fill_between(
        results_test['y_true'],
        results_test['mean'] - results_test['std'],
        results_test['mean'] + results_test['std'],
        alpha=0.2,
        color='orange'
    )
    
    ax[1,2].fill_between(
        results_test_permuted['y_true'],
        results_test_permuted['mean'] - results_test_permuted['std'],
        results_test_permuted['mean'] + results_test_permuted['std'],
        alpha=0.2,
        color='orange'
    )
    
   
    ax[1,0].set_xlabel("delta true")
    ax[1,1].set_xlabel("delta true")
    ax[1,0].set_ylabel("delta pred")
    ax[1,0].legend()
    ax[1,1].legend()

    ax[3, 0].plot(history_train_reg[20:], label = 'train')
    ax[3, 0].plot(history_val_reg[20:],   label = 'val')
    
    ax[3, 1].plot(history_train_recon[20:])
    ax[3, 1].plot(history_val_recon[20:])
    
    ax[3, 2].plot(history_train_kl[20:])
    ax[3, 2].plot(history_val_kl[20:])
  
    
    ax[3,0].set_title("Regression loss")
    ax[3,1].set_title("Reconstruction loss")
    ax[3,2].set_title("KL loss")
  
    ax[3,0].legend()    
    
 
    path = "./results/"
    filename = "fig_predictions_data_transformation.{:d}_".format(on_off_transformation) + "_" + str(configuration_key  ) + "_" + latent_regressor_architecture
    suffix = ".png"
    plt.savefig(path + filename + suffix, dpi = 500, format = 'png')
    plt.show()