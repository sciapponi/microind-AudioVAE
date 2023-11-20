import torch
import torch.nn as nn
from torch import Tensor
import torch.functional as F
from typing import *

class Encoder(nn.Module):

    def __init__(self,
                 num_classes,
                 latent_dim,
                 hidden_dims: List = None,
                 spec_time: int = 64,
                 spec_bins: int = 64) ->None:
        
        in_channels = 1 # spectrogram input channels
        self.spec_time = spec_time
        self.spec_bins = spec_bins

        # embed class for input conditioning and data embedding
        self.embed_class = nn.Linear(num_classes, spec_bins * spec_time)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        encoder_modules = []

        in_channels += 1 # To account for the extra label channel to add conditioning

        #Build Encoder
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        #Encoder unpacking, mu and var declaration
        self.encoder = nn.Sequential(*encoder_modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def forward(self, batch):
        y = batch[1].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.spec_time, self.spec_bins).unsqueeze(1) # check wheter to swap time and bins
        
        embedded_input = embedded_input(batch[0])
        
        mu, log_var = self.encode(batch[0])

        z = self.reparameterize(mu, log_var)
        
        return [z, y, mu, log_var]


class Decoder(nn.Module):
    def __init__(self,
                 num_classes,
                 latent_dim,
                 hidden_dims: List = None) ->None:
        
        modules = []

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU())

        self.head = nn.Sequential(
                                    nn.Conv2d(hidden_dims[-1], 
                                              out_channels= 3,
                                              kernel_size= 3, 
                                              padding= 1),
                                    nn.Tanh())



        self.decoder = nn.Sequential(*modules)
        
    def decode(self, z: Tensor) -> Tensor:
        
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = self.head(result)
        return result
    
    def forward(self, z, y):
        z = torch.cat([z, y], dim = 1)
        result = self.decode(z)
        return result



