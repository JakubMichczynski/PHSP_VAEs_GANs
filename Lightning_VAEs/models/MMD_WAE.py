import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor

#Check https://github.com/tolstikhin/wae and https://github.com/AntixK/PyTorch-VAE


class Lit_MMD_WAE(pl.LightningModule):
    # kernel_mul=2.0, kernel_num=5, fix_sigma=None
    def __init__(self, learning_rate=0.0005, name='MMD_WAE', mmd_weight=50,  kernel_type='imq', lantent_var=2.0, scheduler_gamma=0.95, weight_decay=0.0):
        super(Lit_MMD_WAE, self).__init__()

        self.name = name
        self.learning_rate = learning_rate
        self.mmd_weight = mmd_weight
        self.kernel_type = kernel_type
        self.lantent_var = lantent_var
        self.scheduler_gamma = scheduler_gamma
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(6, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            torch.nn.Linear(200, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            nn.Linear(400, 400),
            nn.BatchNorm1d(400), 
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200), 
            nn.LeakyReLU(),
            nn.Linear(200, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 6)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def mmd_wae_loss_function(self, decoded, input, encoded) -> dict:

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        mmd_weight = self.mmd_weight/bias_corr

        recons_loss = F.mse_loss(decoded, input)
        mmd_loss = self.compute_mmd(encoded, mmd_weight)

        combined_loss = recons_loss + mmd_loss

        return combined_loss, recons_loss, mmd_loss

    def compute_radial_basis_function(self, x1: Tensor, x2: Tensor) -> Tensor:

        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.lantent_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inverse_multiquadric_kernel(self, x1: Tensor, x2: Tensor, eps: float = 1e-7) -> Tensor:

        z_dim = x2.size(-1)
        C = 2 * z_dim * self.lantent_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-3)

        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_radial_basis_function(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inverse_multiquadric_kernel(x1, x2)
        else:
            raise ValueError('Undefined kernel')

        return result

    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from normal distribution
        prior_z = torch.randn_like(z, device=self.device)

        prior_z_prior_z_kernel = self.compute_kernel(prior_z, prior_z)
        z_z_kernel = self.compute_kernel(z, z)
        prior_z_z_kernel = self.compute_kernel(prior_z, z)

        mmd =  z_z_kernel.mean() + prior_z_prior_z_kernel.mean()- 2 * prior_z_z_kernel.mean()
        return mmd

    def training_step(self, batch, batch_idx):
        features = batch
        encoded, decoded = self(features)

        combined_loss, mse_loss, mmd_loss = self.mmd_wae_loss_function(
            decoded=decoded, encoded=encoded, input=batch)

        train_logs = {"combined_loss": combined_loss.detach(
        ), "mse_loss": mse_loss.detach(), "mmd_loss": mmd_loss.detach()}
        self.log("train_logs", train_logs, on_step=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        features = batch

        encoded, decoded = self(features)

        combined_loss, mse_loss, mmd_loss = self.mmdwae_loss_function(
            decoded=decoded, encoded=encoded, input=batch)

        validation_logs = {"combined_loss": combined_loss.detach(
        ), "mse_loss": mse_loss.detach(), "mmd_loss": mmd_loss.detach()}
        self.log("validation_logs", validation_logs)
        return combined_loss.detach()

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        val_epoch_end_logs = {"avg_val_loss": avg_loss.detach()}
        self.log("validation_epoch_end_logs", val_epoch_end_logs)
        return avg_loss.detach()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]
