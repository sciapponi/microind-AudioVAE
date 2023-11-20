from micromind import MicroMind
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from AudioVAE import Encoder, Decoder
from dataset import AudioMNIST
from torch.utils.data import DataLoader

class AudioVAE(MicroMind):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # Extractor parameters
        scale = 1
        sample_rate = int(48000 * scale)
        hop_size = int(320 * scale)
        mel_bins = 64
        window_size = int(1024 * scale)
        fmin = 50
        fmax = 14000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.modules["spectrogram_extractor"] = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
                        freeze_parameters=True)

        # Logmel feature extractor
        self.modules["logmel_extractor"] = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        
        # add Encoder to modules
        self.modules["encoder"] = Encoder(num_classes=10, 
                                            latent_dim=128,
                                            hidden_dims=[32, 64, 128],
                                            spec_time=150,
                                            spec_bins=64)
        # add Decoder to Modules
        self.modules["decoder"] = Decoder(num_classes=10,
                                          latent_dim=128,
                                          hidden_dims=[32, 64, 128])
        
        def forward(self, batch):
            # RESIZE IMAGES (COuld use collate_fn)
            x = self.modules["spectrogram_extractor"](batch[0])
            x = self.modeules["logmel_extractor"](x)
            z, y, mu, log_var = self.modules["encoder"](x)
            return x
        

        

if __name__=="__main__":
    scale = 1
    sample_rate = int(44100 * scale)
    n_fft = 1028
    hop_size = int(320 * scale)
    mel_bins = 64
    window_size = int(1024 * scale)
    fmin = 50
    fmax = 14000
    duration = 5
    crop_len = 5 * sample_rate
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    
    base_path = '/home/ste/Code/python/micromind-warmup-task/task_2/AudioMNIST/data'
    
    dataset = AudioMNIST(base_path)

    waveform, sr, label, speaker = dataset[0]

    spec_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
                        freeze_parameters=True)
    
    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
    
    
    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        print(batch[0].squeeze().shape) # removes all axis with dim=1 (removes channels)
        spec = spec_extractor(batch[0].squeeze())
        print(spec.shape)
        batch[0] = logmel_extractor(spec)
        print(batch[0].shape)
        encoder = Encoder(num_classes=10, 
                        latent_dim=32,
                        hidden_dims=[32, 64, 128],
                        spec_time=150,
                        spec_bins=64)
        
        
        
        print(encoder(batch))
        exit()
    