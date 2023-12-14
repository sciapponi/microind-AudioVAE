import torch
from torch import nn 
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank, STFT
import numpy as np
from typing import List
from torchvision.utils import save_image

class STFTPhaseWrapper(nn.Module):
    def __init__(self,
                n_fft, 
                hop_length,
                win_length,
                eps: float = 1e-8,
                window="hann_window"):
        
        super().__init__()
        self.fft_size = n_fft
        self.hop_size = hop_length
        self.win_length = win_length
        self.window = window
        self.eps = eps

    def forward(self, x):
        x_stft = torch.stft(
            input = x,
            n_fft = self.fft_size,
            hop_length = self.hop_size,
            win_length = self.win_length,
            # window = self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=self.eps)
        )
        x_phs = torch.angle(x_stft)
        return x_mag, x_phs
        

class MultiResolutionSpecLoss(nn.Module):

    def __init__(self,
                sample_rate = 22050,
                mel_bins = 80,
                fft_sizes: List[int] = [1024, 2048, 512],
                hop_sizes: List[int] = [120, 240, 50],
                win_lengths: List[int] = [600, 1200, 240],
                center=True,
                pad_mode = 'reflect',
                window: str = "hann",
                fmin = 0,
                fmax = 8000,
                ref = 1.0,
                amin = 1e-10,
                top_db = None,
                w_sc: float = 1.0,
                w_log_mag: float = 1.0,
                w_lin_mag: float = 0.0,
                w_phs: float = 0.0,
                scale: str = None,
                n_bins: int = None,
                perceptual_weighting: bool = False,
                scale_invariance: bool = False,):

        super().__init__()

        self.stfts = []
        self.spec_extractors = []
        self.logmel_extractors = []

        for i in range((len(fft_sizes))):

            # stft = STFT(n_fft=fft_sizes[i], 
            #             hop_length=hop_sizes[i],
            #             win_length=win_lengths[i],
            #             center=center,
            #             pad_mode=pad_mode,
            #             freeze_parameters=True)

            stft = STFTPhaseWrapper(n_fft=fft_sizes[i],
                                    hop_length=hop_sizes[i],
                                    win_length=win_lengths[i])

            
            spec_extractor = Spectrogram(n_fft=fft_sizes[i], 
                                        hop_length=hop_sizes[i],
                                        win_length=win_lengths[i],
                                        window=window, 
                                        center=center, 
                                        pad_mode=pad_mode,
                                        freeze_parameters=True)
            
            log_mel_extractor = LogmelFilterBank(sr=sample_rate, 
                                                n_fft=fft_sizes[i],
                                                n_mels=mel_bins, 
                                                fmin=fmin, 
                                                fmax=fmax, 
                                                ref=ref, 
                                                amin=amin, 
                                                top_db=top_db, 
                                                freeze_parameters=True)
            
            self.stfts.append(stft)
            self.spec_extractors.append(spec_extractor)
            self.logmel_extractors.append(log_mel_extractor)
        
        self.stfs = nn.ModuleList(self.stfts)
        self.spec_extractors = nn.ModuleList(self.spec_extractors)
        self.logmel_extractors = nn.ModuleList(self.logmel_extractors)

    
    def phase_loss(self, x_phs, y_phs):
        # x_stft =  torch.cat(x_stft)
        # print(x_stft[1])
        # x_phase_angles = self.unwrap(torch.angle(torch.view_as_complex(torch.stack(x_stft,dim=-1))))
        # y_phase_angles = self.unwrap(torch.angle(torch.view_as_complex(torch.stack(y_stft,dim=-1))))
        # x_phase = self.unwrap(torch.atan2(x_stft[0], x_stft[1]))
        # print(x_phase)
        x_phs = torch.remainder(x_phs, 2*np.pi)/(2*np.pi)
        y_phs = torch.remainder(y_phs, 2*np.pi)/(2*np.pi)
        return F.mse_loss(x_phs, y_phs)
    
    def spectral_convergence_loss(self, x_spec, y_spec):

        spec_loss_sc = torch.norm(torch.abs(y_spec) - torch.abs(x_spec)) / (torch.norm(torch.abs(y_spec)) + 1e-9)
        # print(spec_loss_sc)
        return spec_loss_sc

    def mel_reconstruction_loss(self, x_mel, y_mel):
        mel_loss = F.l1_loss(x_mel, y_mel)
        # print(mel_loss)
        return mel_loss
    
    def forward(self, pred, batch):
        length_mask = batch[2]
        recons = pred[0]*length_mask
        input = pred[1]*length_mask

        cumulative_loss = 0.0

        for stft, spec_extractor, logmel_extractor in zip(self.stfts, self.spec_extractors, self.logmel_extractors):
            
            # print(input.shape)
            input_mag, input_phs = stft(input.squeeze())
            recons_mag, recons_phs = stft(recons.squeeze())
            
            phase_loss = self.phase_loss(input_phs, recons_phs)

            input_spec = spec_extractor(input.squeeze())
            recons_spec = spec_extractor(recons.squeeze())

            spectral_convergence_loss = self.spectral_convergence_loss(input_spec, recons_spec)

            input_mel =  logmel_extractor(input_spec)
            recons_mel = logmel_extractor(recons_spec)

            mel_reconstruction_loss  = self.mel_reconstruction_loss(input_mel, recons_mel)
            print("phase loss: ", phase_loss, "sc loss",spectral_convergence_loss, "mel_loss", mel_reconstruction_loss )
            cumulative_loss += 0.35*phase_loss + 0.15* spectral_convergence_loss + 0.5* mel_reconstruction_loss

            
        cumulative_loss /= len(self.stfts)

        return cumulative_loss


        
        


                