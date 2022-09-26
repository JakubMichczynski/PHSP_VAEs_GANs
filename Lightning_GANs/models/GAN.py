import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generate = nn.Sequential(
            nn.Linear(self.latent_dim, 400),
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
            torch.nn.Linear(400, 6)
        )

    def forward(self, x):
        return self.generate(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.discriminate = nn.Sequential(
            nn.Linear(6, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x)


class Lit_GAN(pl.LightningModule):
    def __init__(self, name: str = 'GAN', latent_dim: int = 6, learning_rate: float = 0.0005, b1: float = 0.5, b2: float = 0.999, weight_decay: float = 0.0, scheduler_gamma: float = 0.95):
        super(Lit_GAN, self).__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.latent_dim)
        self.discriminator = Discriminator()

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        photons_batch = batch
        # Forward pass
        # sample noise
        noise = torch.randn(photons_batch.shape[0], self.latent_dim)
        noise = noise.type_as(photons_batch)

        # train generator
        if optimizer_idx == 0:

            # generate photons
            self.generated_photons = self(noise)

            valid = torch.ones(photons_batch.size(0), 1)
            valid = valid.type_as(photons_batch)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(
                self.discriminator(self(noise)), valid)
            train_logs = {"g_loss": g_loss.detach()}

            self.log("train_g_logs", train_logs, on_step=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:

            valid = torch.ones(photons_batch.size(0), 1)
            valid = valid.type_as(photons_batch)

            real_loss = self.adversarial_loss(
                self.discriminator(photons_batch), valid)

            fake = torch.zeros(photons_batch.size(0), 1)
            fake = fake.type_as(photons_batch)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(noise).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            train_logs = {"d_loss": d_loss.detach()}

            self.log("train_d_logs", train_logs, on_step=True)
            return d_loss


    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(
        ), lr=self.learning_rate, weight_decay=self.weight_decay, betas=(self.b1, self.b2))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(
        ), lr=self.learning_rate, weight_decay=self.weight_decay, betas=(self.b1, self.b2))
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_g, gamma=self.scheduler_gamma)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_d, gamma=self.scheduler_gamma)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
