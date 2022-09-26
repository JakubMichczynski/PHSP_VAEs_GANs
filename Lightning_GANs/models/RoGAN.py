import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.map1_en = nn.Sequential(
            torch.nn.Linear(self.latent_dim, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
        )
        self.map1_au = nn.Sequential(
            torch.nn.Linear(6, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
        )
        self.maps_enc = nn.Sequential(
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400)
        )
        self.maps_au = nn.Sequential(
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400)
        )
        self.maps_dec = nn.Sequential(
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
            torch.nn.Linear(400, 200),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(400),
            torch.nn.Linear(200, 6)
        )
        # initialisation
        for p in self.parameters():
            if p.ndimension()>1:
                nn.init.kaiming_normal_(p)
                #nn.init.xavier_normal_(p)
                #nn.init.kaiming_uniform_(p, nonlinearity='sigmoid')



    def forward(self, noise, photons_batch):
        photons_batch = self.map1_au(photons_batch)
        noise = self.map1_en(noise)
        
        photons_batch= self.maps_au(photons_batch)
        noise = self.maps_enc(noise)

        latent_decoded_photons = photons_batch
        latent_generated_photons = noise

        decoded_photons = self.maps_dec(photons_batch)
        generated_photons = self.maps_dec(noise)

        return generated_photons, decoded_photons, latent_generated_photons, latent_decoded_photons


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.discriminate = nn.Sequential(
            nn.Linear(6, 200),
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
            nn.Linear(200, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x)


class Lit_RoGAN(pl.LightningModule):
    def __init__(self, name: str = 'RoGAN', latent_dim: int = 6, autoencoder_weight: float = 1. , latent_weight: float = 1. , learning_rate: float = 0.0005, critic_iterations: int = 5, weight_clip: float = 0.01, weight_decay: float = 0.0, scheduler_gamma: float = 0.95):
        super(Lit_RoGAN, self).__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.autoencoder_weight = autoencoder_weight
        self.latent_weight = latent_weight
        self.learning_rate = learning_rate
        self.critic_iterations = critic_iterations
        self.weight_clip = weight_clip
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.latent_dim)
        self.critic = Critic()

    def forward(self, noise, photons_batch):
        return self.generator(noise, photons_batch)


    def generator_step(self, noise, photons_batch):
        generated_photons, decoded_photons, latent_generated_photons, latent_decoded_photons = self(noise, photons_batch)
        g_loss = -torch.mean(self.critic(generated_photons))
        g_loss += self.autoencoder_weight * F.mse_loss(photons_batch,decoded_photons,reduction='mean')
        g_loss += self.latent_weight * F.mse_loss(latent_generated_photons, latent_decoded_photons, reduction='mean')

        train_logs = {"g_loss": g_loss.detach()}

        self.log("train_g_logs", train_logs, on_step=True, prog_bar=True)
        return g_loss


    def critic_step(self, noise, photons_batch):
        generated_photons,_,_,_ = self(noise, photons_batch)
        critic_real = self.critic(photons_batch)
        critic_fake = self.critic(generated_photons)
        critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))

        # for p in self.critic.parameters():
        #         p.data.clamp_(-self.weight_clip, self.weight_clip)

        train_logs = {"c_loss": critic_loss.detach()}

        self.log("train_c_logs", train_logs, on_step=True, prog_bar=True)
        return critic_loss


    def training_step(self, batch, batch_idx, optimizer_idx):
        photons_batch = batch

        # Forward pass
        # sample noise
        noise = torch.randn(photons_batch.shape[0], self.latent_dim)
        noise = noise.type_as(photons_batch)

        # train discriminator
        if optimizer_idx == 1:
            return self.critic_step(noise=noise, photons_batch=photons_batch)

        # train generator
        if optimizer_idx == 0:
            return self.generator_step(noise=noise, photons_batch=photons_batch)


    def configure_optimizers(self):
        optimizer_g = torch.optim.RMSprop(self.generator.parameters(
        ), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer_c = torch.optim.RMSprop(self.critic.parameters(
        ), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_g, gamma=self.scheduler_gamma)
        scheduler_c = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_c, gamma=self.scheduler_gamma)
        return [optimizer_g, optimizer_c], [scheduler_g, scheduler_c]

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False):

        # update critic every step
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)
            for p in self.critic.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)

        # update generator every "critic_iterations" steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.critic_iterations == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
