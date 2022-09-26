import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import Tensor

#Check https://github.com/AntixK/PyTorch-VAE


class Lit_InfoVAE(pl.LightningModule):
    def __init__(self, learning_rate=0.0005, name='InfoVAE', mmd_weight=20,  kernel_type='imq', kld_weight=-2.0, reconstruction_weight=4.5, bias_correction_term=0.00131, lantent_var=2.0, weight_decay=0.0, scheduler_gamma=0.95):
        super(Lit_InfoVAE, self).__init__()

        self.name = name
        self.learning_rate = learning_rate
        self.mmd_weight = mmd_weight
        self.kernel_type = kernel_type
        self.kld_weight = kld_weight
        self.reconstruction_weight = reconstruction_weight
        self.bias_correction_term = bias_correction_term  # batch_size/num_of_photons
        self.lantent_var = lantent_var
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma

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
        )

        self.z_mean = nn.Sequential(
            nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Linear(200, 4)

        )

        self.z_log_var = nn.Sequential(
            nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(),
            nn.Linear(200, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 200),
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

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1), device=self.device)
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded


    def info_loss_function(self, decoded, input, encoded, z_mean, z_log_var):

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)

        recons_loss = F.mse_loss(decoded, input)
        mmd_loss = self.compute_mmd(encoded)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var -
                              z_mean ** 2 - z_log_var.exp(), dim=1), dim=0)

        combined_loss = self.reconstruction_weight * recons_loss + \
            (1. - self.kld_weight) * self.bias_correction_term * kld_loss + \
            (self.kld_weight + self.mmd_weight - 1.)/bias_corr * mmd_loss

        return combined_loss, recons_loss, mmd_loss, -kld_loss


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

    def training_step(self, batch):
        features = batch
        # Forward pass
        encoded, z_mean, z_log_var, decoded = self(features)
        combined_loss, mse_loss, mmd_loss, kld_loss = self.info_loss_function(
            decoded=decoded, encoded=encoded, z_mean=z_mean, z_log_var=z_log_var, input=batch)

        train_logs = {"combined_loss": combined_loss.detach(), "mse_loss": mse_loss.detach(
        ), "mmd_loss": mmd_loss.detach(), "kld_loss": kld_loss.detach()}

        self.log("train_logs", train_logs, on_step=True)
        return combined_loss

    def validation_step(self, batch):
        features = batch

        # Forward pass
        encoded, z_mean, z_log_var, decoded = self(features)

        combined_loss, mse_loss, mmd_loss, kld_loss = self.info_loss_function(
            decoded=decoded, encoded=encoded, z_mean=z_mean, z_log_var=z_log_var, input=batch)

        validation_logs = {"combined_loss": combined_loss.detach(), "mse_loss": mse_loss.detach(
        ), "mmd_loss": mmd_loss.detach(), "kld_loss": kld_loss.detach()}
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
