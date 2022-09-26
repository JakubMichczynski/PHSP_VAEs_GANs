import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import itertools

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.discriminate = nn.Sequential(
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
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x)

class Lit_AAE(pl.LightningModule):
    def __init__(self, name: str = 'AAE', latent_dim: int = 6, learning_rate: float = 0.0001, beta_weight: float = 1, discriminator_iterations: int = 5, b1: float = 0.0, b2: float = 0.9, weight_decay: float = 0.0, scheduler_gamma: float = 0.95, bias_correction_term=0.00131):
        super(Lit_AAE, self).__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.beta_weight = beta_weight
        # self.discriminator_iterations = discriminator_iterations
        self.b1 = b1
        self.b2 = b2

        self.bias_correction_term = bias_correction_term
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.save_hyperparameters()


        self.discriminator=Discriminator(self.latent_dim)

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

        self.automatic_optimization = False

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1), device=self.device)
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

    def sample_prior(self, batch_size):
        samples = torch.randn(batch_size, self.latent_dim)
        return samples.to(self.device)

    def adversarial_loss(self, pred, target_is_real=True):
            if target_is_real:
                target = torch.ones_like(pred)
            else:
                target = torch.zeros_like(pred)
            return F.binary_cross_entropy_with_logits(pred, target)

    def training_step(self, batch, batch_idx):
        photons_batch = batch  # (N, C, H, W)
        batch_size = photons_batch.shape[0]
        opt_g, opt_d = self.optimizers()

        # reconstruction phase
        q_z = self.encoder(photons_batch) # (N, hidden_dim)
        generated_photons = self.decoder(q_z)
        recon_loss = F.mse_loss(photons_batch, generated_photons)

        self.log("train_loss/recon_loss", recon_loss)
        opt_g.zero_grad()
        self.manual_backward(recon_loss*self.beta_weight)
        opt_g.step()

        # update discriminator
        real_prior = self.sample_prior(batch_size)
        real_logit = self.discriminator(real_prior)
        real_loss = self.adversarial_loss(real_logit, True)
        fake_logit = self.discriminator(self.encoder(photons_batch))
        fake_loss = self.adversarial_loss(fake_logit, False)
        d_adv_loss = (real_loss + fake_loss) / 2
        self.log("train_loss/d_loss", d_adv_loss)
        self.log("train_log/real_logit", real_logit.mean())
        self.log("train_log/fake_logit", fake_logit.mean())

        opt_d.zero_grad()
        self.manual_backward(d_adv_loss)
        opt_d.step()

        # update generator
        q_z = self.encoder(photons_batch)
        g_adv_loss = self.adversarial_loss(self.discriminator(q_z), True)
        self.log("train_loss/adv_encoder_loss", g_adv_loss)

        opt_g.zero_grad()
        self.manual_backward(g_adv_loss)
        opt_g.step()

        train_logs = {"adv_encoder_loss": g_adv_loss.detach(
        ), "recon_loss": recon_loss.detach(), "d_loss": d_adv_loss.detach(), "real_logit": real_logit.mean(), "fake_logit": fake_logit.mean()}

        self.log("train_logs", train_logs, on_step=True)

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam( itertools.chain(self.encoder.parameters(),self.decoder.parameters()), lr=self.learning_rate, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        optimizer_c = torch.optim.Adam(self.discriminator.parameters(
        ), lr=self.learning_rate, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        return [optimizer_g, optimizer_c]
