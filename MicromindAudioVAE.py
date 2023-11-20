from micromind import MicroMind
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from AudioVAE import Encoder, Decoder
from dataset import AudioMNIST

class AudioVAE(MicroMind):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                                            hidden_dims=[32, 64, 128])
        self.modules["decoder"] = Decoder(num_classes=10,
                                          latent_dim=128,
                                          hidden_dims=[32, 64, 128])
        

        # add Decoder to Modules

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
    
    

    spec = spec_extractor(waveform)
    logmel_spec = logmel_extractor(spec)
    print(spec.shape)
    print(logmel_spec.shape)