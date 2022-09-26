import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generate = nn.Sequential(
            nn.Linear(self.latent_dim, 200),
            nn.LayerNorm(200),
            nn.LeakyReLU(),
            torch.nn.Linear(200, 400),
            nn.LayerNorm(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LayerNorm(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LayerNorm(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LayerNorm(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LayerNorm(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.LayerNorm(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 200),
            nn.LayerNorm(200),
            nn.LeakyReLU(),
            torch.nn.Linear(200, 6)
        )

    def forward(self, x):
        return self.generate(x)


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


class Lit_WGAN_gp(pl.LightningModule):
    def __init__(self, name: str = 'WGAN_gp', latent_dim: int = 6, learning_rate: float = 0.0001, critic_iterations: int = 5, gp_param: float = 10, b1: float = 0.0, b2: float = 0.9, weight_decay: float = 0.0, scheduler_gamma: float = 0.95):
        super(Lit_WGAN_gp, self).__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.critic_iterations = critic_iterations
        self.gp_param = gp_param
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.latent_dim)
        self.critic = Critic()

    def forward(self, x):
        return self.generator(x)


    def generator_step(self, noise):
        g_loss = -torch.mean(self.critic(self(noise)))

        train_logs = {"g_loss": g_loss.detach()}

        self.log("train_g_logs", train_logs, on_step=True, prog_bar=True)
        return g_loss


    def critic_step(self, noise, photons_batch):
        fake_photons = self(noise)
        critic_real = self.critic(photons_batch)
        critic_fake = self.critic(fake_photons)
        gradient_penalty = self.gradient_penalty(
            critic=self.critic, photons_batch=photons_batch, fake_photons=fake_photons)
        critic_loss = -(torch.mean(critic_real) -
                        torch.mean(critic_fake)) + self.gp_param * gradient_penalty

        train_logs = {"c_loss": critic_loss.detach(
        ), "gradient_penalty": gradient_penalty.detach()}

        self.log("train_c_logs", train_logs, on_step=True, prog_bar=True)
        return critic_loss

    def get_interpolation(self, fake_photons, photons_batch):
        alpha = torch.rand(size=(photons_batch.size(0), 1), device=self.device)
        interpolated_photons = photons_batch * \
            alpha + fake_photons * (1 - alpha)
        return interpolated_photons

    def gradient_penalty(self, critic, photons_batch, fake_photons):
        interpolated_photons = self.get_interpolation(
            fake_photons=fake_photons, photons_batch=photons_batch)
        interpolated_photons.requires_grad = True

        mixed_scores = critic(interpolated_photons)

        gradient = torch.autograd.grad(
            inputs=interpolated_photons,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,)[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

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
            return self.generator_step(noise=noise)


    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(
        ), lr=self.learning_rate, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        optimizer_c = torch.optim.Adam(self.critic.parameters(
        ), lr=self.learning_rate, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
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

        # update generator every 4 steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.critic_iterations == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()
