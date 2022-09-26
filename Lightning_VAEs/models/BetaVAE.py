import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Lit_BetaVAE(pl.LightningModule):
    def __init__(self, learning_rate=0.0005, beta_weight=1, bias_correction_term=0.00131, name='BetaVAE', weight_decay=0.0, scheduler_gamma=0.95):
        super(Lit_BetaVAE, self).__init__()
        self.name = name
        self.learning_rate = learning_rate
        self.beta_weight = beta_weight
        self.bias_correction_term = bias_correction_term
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(6, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
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
            nn.LeakyReLU(),
            nn.LayerNorm(400),
            nn.Linear(400, 100),
            nn.LeakyReLU(),
            nn.LayerNorm(100),
            nn.Linear(100, 12),

        )

        self.z_log_var = nn.Sequential(
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            nn.LayerNorm(400),
            nn.Linear(400, 100),
            nn.LeakyReLU(),
            nn.LayerNorm(100),
            nn.Linear(100, 12),
        )

        self.decoder = nn.Sequential(
            nn.Linear(12, 100),
            nn.ReLU(),
            nn.Linear(100, 400),
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
            nn.Linear(400, 6)
        )

        # initialisation
        for p in self.parameters():
            if p.ndimension()>1:
                nn.init.kaiming_normal_(p)
                #nn.init.xavier_normal_(p)
                #nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')

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

    def training_step(self, batch, _):
        features = batch

        _, z_mean, z_log_var, decoded = self(features)
        kld_loss = -0.5 * torch.sum(1 + z_log_var -
                                    z_mean**2 - torch.exp(z_log_var), axis=1)
        batchsize = kld_loss.size(0)
        kld_loss = kld_loss.mean()

        mse_loss = F.mse_loss(features, decoded, reduction='none')
        mse_loss = mse_loss.view(batchsize, -1).sum(axis=1)
        mse_loss = mse_loss.mean()

        combined_loss = mse_loss+self.beta_weight*self.bias_correction_term*kld_loss

        train_logs = {"combined_loss": combined_loss.detach(
        ), "mse_loss": mse_loss.detach(), "kld_loss": kld_loss.detach()}

        self.log("train_logs", train_logs, on_step=True)
        return combined_loss

    def validation_step(self, batch, _):
        features = batch

        _, z_mean, z_log_var, decoded = self(features)

        kld_loss = -0.5 * torch.sum(1 + z_log_var -
                                    z_mean**2 - torch.exp(z_log_var), axis=1)
        batchsize = kld_loss.size(0)
        kld_loss = kld_loss.mean()

        mse_loss = F.mse_loss(features, decoded, reduction='none')
        mse_loss = mse_loss.view(batchsize, -1).sum(axis=1)
        mse_loss = mse_loss.mean()

        combined_loss = mse_loss+self.beta_weight*kld_loss

        validation_logs = {"combined_loss": combined_loss.detach(
        ), "mse_loss": mse_loss.detach(), "kld_loss": kld_loss.detach()}
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
