import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import *
from torch.nn.utils.parametrizations import weight_norm

class Encoder(nn.Module):
    def __init__(self,
                 num_classes,
                 latent_dim,
                 hidden_dims: List = None,
                 spec_time: int = 64,
                 spec_bins: int = 64,
                 in_channels = 1 # spectrogram input channels
                 ) ->None:
        super(Encoder, self).__init__()

        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.spec_time = spec_time
        self.spec_bins = spec_bins
        self.latent_dim = latent_dim
        # embed class for input conditioning and data embedding
        self.embed_class = nn.Linear(num_classes, spec_bins * spec_time)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        encoder_modules = []

        in_channels += 1 # To account for the extra label channel to add conditioning

        #Build Encoder
        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    weight_norm(nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1)),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        #Encoder unpacking, mu and var declaration
        self.encoder = nn.Sequential(*encoder_modules)
        # self.fc_mu = nn.Linear(hidden_dims[-1]*12*10, latent_dim) # < depends on downsampling
        # self.fc_var = nn.Linear(hidden_dims[-1]*12*10, latent_dim)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def encode(self, input: Tensor) -> List[Tensor]:

        result = self.encoder(input)
        bs, _, h, w = result.shape
        result = result.reshape(-1, self.hidden_dims[-1])
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
   
        mu = self.fc_mu(result).reshape(bs, self.latent_dim, h, w)
        # print(mu.shape)
        log_var = self.fc_var(result).reshape(bs, self.latent_dim, h, w)

        return [mu, log_var]
    
    def forward(self, batch, y):

        y = F.one_hot(y.long(), self.num_classes).float()

        embedded_class = self.embed_class(y)

        # print("A",embedded_class.shape)
        embedded_class = embedded_class.view(-1,  self.spec_time, self.spec_bins).unsqueeze(1) # check wheter to swap time and bins
        # print("A",embedded_class.shape)
        # print(batch.shape)
        embedded_input = self.embed_data(batch)
        # print("B",embedded_input.shape)
        x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        return [z, y, mu, log_var]


class SpecDecoder(nn.Module):
    def __init__(self,
                 num_classes,
                 latent_dim,
                 spec_bins,
                 spec_time,
                 hidden_dims: List = None) ->None:
        super(SpecDecoder, self).__init__()
        modules = []

        self.spec_time = spec_time
        self.spec_bins = spec_bins
        # self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1]*12*10) #hardcoded
        self.embed_class = nn.Linear(num_classes, spec_bins * spec_time)
        hidden_dims.append(latent_dim+1)
        hidden_dims.reverse()
        self.hidden_dims = hidden_dims
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
                                               stride=1,
                                               padding=(2,1),
                                               ),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU())

        self.head = nn.Sequential(nn.Conv2d(hidden_dims[-1], 
                                              out_channels= 1,
                                              kernel_size= 3, 
                                              padding= 1),
                                    # nn.Tanh())
                                    nn.Sigmoid())




        self.decoder = nn.Sequential(*modules)
        
    def decode(self, z: Tensor) -> Tensor:
        # result = self.decoder_input(z)
        # print(result.shape)
        # result = result.view(-1, self.hidden_dims[0],12, 10) #hardcoded 19 x 8
        result = self.decoder(z)
        # print(f"decoder ouput",result.shape)
        # print(result.shape)
        result = self.final_layer(result)
        # print(f"flayer ouput",result.shape)
        result = self.head(result)
        # print(f"head ouput",result.shape)
        return result
    
    def forward(self, z, y):
        y = self.embed_class(y).reshape(z.shape[0], 1, self.spec_time, self.spec_bins)
        z = torch.cat([z, y], dim = 1)
        result = self.decode(z)
        return result


class WaveformConvBlock(nn.Module):
    def __init__(self, channels, kernels):
        super(WaveformConvBlock, self).__init__()
        self.blocks = nn.ModuleList(
            [weight_norm(nn.Conv1d(channels, channels, kernel_size, stride=1, padding=0, dilation=1)) for kernel_size in kernels]
        )
    def forward(self, x):
        for resblock in self.blocks:
            x = resblock(x)
        return x

class WaveformDecoder(nn.Module):

    def __init__(self,
                 num_classes,
                 spec_time, 
                 spec_bins,
                 n_channels = 512,
                #  upsample_kernel_sizes= [16,16,4,4,4],
                #  upsample_rates = [16,16,4,4, 2],
                #  latent_dim = 32,
                #  conv_kernel_sizes = [3, 3, 3, 7, 11]):
                upsample_kernel_sizes= [3,3],
                 upsample_rates = [32,32],
                 latent_dim = 32,
                 conv_kernel_sizes = [3, 3, 3]):

        # Takes a generated latent representation of a spectrogram as an input (sampled from VAE latent space) 
        # Outputs waveform

        super(WaveformDecoder, self).__init__()

        self.spec_time = spec_time
        self.spec_bins = spec_bins
        self.embed_class = nn.Linear(num_classes, spec_bins * spec_time)

        

        self.first = nn.Conv1d(in_channels=spec_bins*(latent_dim+1), out_channels=n_channels, kernel_size=7, stride=1, padding=3)
        
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()

        channels_now = n_channels

        for i in range(len(upsample_kernel_sizes)):
            out_channels = channels_now // 2
            ## just added weight norm
            self.ups += [weight_norm(nn.ConvTranspose1d(channels_now, out_channels, kernel_size=upsample_kernel_sizes[i], stride=upsample_rates[i], padding=0))] 
            self.convs += [WaveformConvBlock(out_channels, conv_kernel_sizes)]
            channels_now = out_channels

        self.last = nn.Conv1d(channels_now, 1, kernel_size=3, stride=1, padding=0)
        
        self.last_upsampling = nn.AdaptiveAvgPool1d(22050) # horrible upsampling to 22050
    
    def decode(self, z: Tensor) -> Tensor:

        x = self.first(z)

        for up, conv in zip(self.ups, self.convs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            x = conv(x)
        print("before lastblock",x[0][0])
        x = F.leaky_relu(x, 0.1)

        x = self.last(x)
        print("last", x[0][0])
        x = self.last_upsampling(x)
        print('fixing size', x[0][0])
        x = torch.tanh(x)
        # x = torch.sigmoid(x)
        return x
    
    def forward(self, z, y):
        # print("y", y.shape)
        y = self.embed_class(y).reshape(z.shape[0], 1, self.spec_bins, self.spec_time)
        # print("y", y.shape)
        # print("z", z.shape)
        z = torch.cat([z, y], dim = 1)
        Bs, channels, bins, time = z.shape
        # print(Bs, channels, bins, time)
        z = z.reshape(Bs, time, bins*channels) # original torchlibrosa chape: batch_size x time x n_mels
        z = z.permute(0,2,1) # permute shape to put frequencies as the second dimension -> batch_size x n_mels x time
        # print(z.shape)
        # print(z[0])
        # exit()
        result = self.decode(z)
        # print(result.shape)
        return result