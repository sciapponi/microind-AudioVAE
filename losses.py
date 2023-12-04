import torch
from torch import nn 
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank, STFT
from types import List

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

            stft = STFT(n_fft=fft_sizes[i], 
                        hop_length=hop_sizes[i],
                        win_length=win_lengths[i],
                        center=center,
                        pad_mode=pad_mode,
                        freeze_parameters=True)
            
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


    def phase_loss(self, x_stft, y_stft):
        x_phase_angles = torch.angle(x_stft[0], x_stft[1])
        y_phase_angles = torch.angle(y_stft[0], y_stft[1])
        return F.mse_loss(x_phase_angles, y_phase_angles)
    
    def spectral_convergence_loss(self, x_spec, y_spec):

        return torch.norm(y_spec - x_spec, p="fro") / torch.norm(y_mag, p="fro")

    def mel_reconstruction_loss(self, x_mel, y_mel):
        return F.l1_loss(x_mel, y_mel)
    
    def forward(self, pred, batch):
        length_mask = batch[2]
        recons = pred[0]*length_mask
        input = pred[1]*length_mask

        cumulative_loss = 0.0

        for stft, spec_extractor, logmel_extractor in zip(self.stfts, self.spec_extractors, self.logmel_extractors):
            
            input_stft = stft(input.squeeze())
            recons_stft = stft(recons.squeeze())
            
            phase_loss = self.phase_loss(input_stft, recons_stft)

            input_spec = spec_extractor(input.squeeze())
            recons_spec = spec_extractor(input.squeeze())

            spectral_convergence_loss = self.spectral_convergence_loss(input_spec, recons_spec)

            input_mel =  logmel_extractor(input_spec)
            recons_mel = logmel_extractor(recons_spec)

            mel_reconstruction_loss  = self.mel_reconstruction_loss(input_mel, recons_mel)

            cumulative_loss += phase_loss + spectral_convergence_loss + mel_reconstruction_loss

            
        cumulative_loss /= len(self.stfts)

        return cumulative_loss


        
        


                