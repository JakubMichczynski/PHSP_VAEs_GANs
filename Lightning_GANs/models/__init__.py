from .GAN import *
from .WGAN import *
from .WGAN_gp import *
from .CGAN import *
from .RoGAN import *


gan_models={'GAN':Lit_GAN,
            'WGAN':Lit_WGAN,
            'WGAN_gp': Lit_WGAN_gp,
            'CGAN': Lit_CGAN,
            'RoGAN': Lit_RoGAN,}