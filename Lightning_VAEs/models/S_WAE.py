import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import Tensor
from torch import distributions as dist

#Check https://github.com/AntixK/PyTorch-VAE and https://github.com/skolouri/swgmm


class Lit_S_WAE(pl.LightningModule):
    def __init__(self, learning_rate=0.0005, name='S_WAE', latent_dim=4, w_weight=110, bias_correction_term=0.00131, wasserstein_deg=2.0,  num_projections=50, projection_dist='normal', weight_decay=0.0, scheduler_gamma=0.95):
        super(Lit_S_WAE, self).__init__()

        self.name = name
        self.learning_rate = learning_rate
        self.w_weight = w_weight
        self.wasserstein_deg = wasserstein_deg
        self.num_projections = num_projections
        self.latent_dim = latent_dim
        self.projection_dist = projection_dist
        self.bias_correction_term = bias_correction_term  # batch_size/num_of_photons
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
            nn.Linear(400, 400),
            nn.BatchNorm1d(400), 
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200), 
            nn.LeakyReLU(),
            nn.Linear(200, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 200),
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
        return encoded, None, None, decoded

    def s_wae_loss_function(self, decoded, input, encoded) -> dict:

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.w_weight / bias_corr

        recons_loss_l2 = F.mse_loss(decoded, input)
        recons_loss_l1 = F.l1_loss(decoded, input)
        recons_loss = recons_loss_l1+recons_loss_l2

        s_wd_loss = self.compute_s_wd(
            encoded=encoded, w_weight=self.w_weight, reg_weight=reg_weight)

        combined_loss = recons_loss + s_wd_loss

        return combined_loss, recons_loss, s_wd_loss

    def get_random_projections(self, latent_dim: int, num_samples: int) -> Tensor:
        if self.projection_dist == 'normal':
            rand_samples = torch.randn(
                num_samples, latent_dim, device=self.device)
        else:
            raise ValueError('Unknown projection distribution.')

        rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
        return rand_proj 

    def compute_s_wd(self,
                     encoded: Tensor,
                     w_weight: float,
                     reg_weight: float) -> Tensor:

        prior_z = torch.randn_like(encoded, device=self.device) 

        proj_matrix = self.get_random_projections(self.latent_dim,
                                                  num_samples=self.num_projections).transpose(0, 1)

        latent_projections = encoded.matmul(proj_matrix)
        prior_projections = prior_z.matmul(proj_matrix)

        w_dist = torch.sort(latent_projections.t(), dim=1)[0] - \
            torch.sort(prior_projections.t(), dim=1)[0]
        w_dist = w_dist.pow(w_weight)
        return reg_weight * w_dist.mean()

    def training_step(self, batch, batch_idx):
        encoded, _, _, decoded = self(batch)

        combined_loss, recons_loss, s_wd_loss = self.s_wae_loss_function(
            decoded=decoded, encoded=encoded, input=batch)

        train_logs = {"combined_loss": combined_loss.detach(
        ), "recons_loss": recons_loss.detach(), "s_wd_loss": s_wd_loss.detach()}
        self.log("train_logs", train_logs, on_step=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        encoded, _, _, decoded = self(batch)

        combined_loss, recons_loss, s_wd_loss = self.s_wae_loss_function(
            decoded=decoded, encoded=encoded, input=batch)

        validation_logs = {"combined_loss": combined_loss.detach(
        ), "recons_loss": recons_loss.detach(), "s_wd_loss": s_wd_loss.detach()}
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
