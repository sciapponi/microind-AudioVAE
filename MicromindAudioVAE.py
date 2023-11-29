from micromind import MicroMind, Metric
from micromind.utils import parse_configuration
import micromind as mm
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchinfo import summary
from AudioVAE import Encoder, SpecDecoder, WaveformDecoder
from dataset import AudioMNIST
import torchaudio
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
import sys

def set_reproducibility(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class AudioVAE(MicroMind):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        #hifigan extractor parameters
        scale = 1
        sample_rate = int(22050 * scale)
        n_fft = 1024
        hop_size = int(256 * scale)
        mel_bins = 80
        window_size = int(1024 * scale)
        fmin = 0
        fmax = 8000
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
        
        self.modules['bn0'] = torch.nn.BatchNorm2d(mel_bins)

        # add Encoder to modules
        self.modules["encoder"] = Encoder(num_classes=10, 
                                            latent_dim=32,
                                            hidden_dims=[32, 64, 128, 256],
                                            spec_time=87,
                                            spec_bins=80)
        # add Decoder to Modules
        # self.modules["decoder"] = SpecDecoder(num_classes=10,
        #                                     latent_dim=32,
        #                                     spec_time=12,
        #                                     spec_bins=10,
        #                                     hidden_dims=[32, 64, 128])
        self.modules["decoder"] = WaveformDecoder(num_classes=10,
                                                    latent_dim=32,
                                                    spec_time=5,
                                                    spec_bins=6
                                                    )
        
        tot_params = 0
        for m in self.modules.values():
            temp = summary(m, verbose=0)
            tot_params += temp.total_params

        # print(self.modules.parameters())
        print(f"Total parameters of model: {tot_params * 1e-6:.2f} M")

    def forward(self, batch):
        # RESIZE IMAGES (COuld use collate_fn
        waveform = batch[0]
        x = self.modules["spectrogram_extractor"](batch[0].squeeze())

        input_mel_spec = self.modules["logmel_extractor"](x)
        x = input_mel_spec.transpose(1, 3)
        x = self.modules['bn0'](x)
        x = x.transpose(1, 3)

        # x = self.modules['bn0'](input_mel_spec)
        z, y, mu, log_var = self.modules["encoder"](x, batch[1])
        print("z",z[0][0])
        x = self.modules["decoder"](z,y)

        return [x, waveform, mu, log_var]

    def kld_loss(self, pred, batch):
        mu = pred[2]
        log_var = pred[3]

        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1))
    
    def recons_loss(self, pred, batch):
        length_mask = batch[2]
        # print("p",pred[0][0])
        print(pred[0].shape)
        print(pred[0][0].mean())
        recons = pred[0]*length_mask # spec artifact removal
        print("r>",recons[0])
        input = pred[1]*length_mask
        print("i>",input[0])
        # print("Pred ",recons[0])
        # print("original ", input[0])
        # waveform_loss = F.l1_loss(recons, input)
        recons = self.modules["logmel_extractor"](self.modules["spectrogram_extractor"](recons.squeeze()))
        input = self.modules["logmel_extractor"](self.modules["spectrogram_extractor"](input.squeeze()))
        spec_loss = F.l1_loss(recons, input)
        return spec_loss
        # return F.l1_loss(recons,input)
        # return waveform_loss
    
    def configure_optimizers(self):
        """Configures the optimizes and, eventually the learning rate scheduler."""
        
        opt = torch.optim.Adam(self.modules.parameters(), lr=0.0000001, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=32900)
        # lr_scheduler = None
        return opt, lr_scheduler
    
    def compute_loss(self, pred, batch):
        # kld_weight = 0.00025
        kld_weight = 0.1
        r_loss = self.recons_loss(pred, batch)
        # print(f"rloss:", r_loss.shape)
        k_loss = self.kld_loss(pred, None)
        # print(r_loss, k_loss)
        # print(f"kloss:", k_loss.shape)
        loss = r_loss + kld_weight * k_loss
        # print(f"loss:", loss.shape)
        # print(loss)
        return loss
        

def scale_waveform(waveform):
    scaled_waveform = 2 * waveform - 1
    return scaled_waveform

if __name__=="__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    set_reproducibility(42)
    checkpointer = mm.utils.checkpointer.Checkpointer(exp_folder, key="loss")
 
    sample_rate = 22050 
    
    base_path = '/home/ste/Code/python/micromind-warmup-task/task_2/AudioMNIST/data'
    
    dataset = AudioMNIST(base_path, resample_rate=sample_rate, max_len_audio=sample_rate)
    # dataset = torch.utils.data.Subset(dataset, [0,1])
    lengths = []
    for waveform, label, length_mask in dataset:
        lengths.append(waveform.shape[1])
    print(f'total {len(set(lengths))}')
    print(set(lengths))

    batch_size = 64
    train_percentage = 0.7
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*train_percentage), int(len(dataset)*(1-train_percentage))])

    train_dataloader = DataLoader(train_set, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    # train_dataloader = DataLoader(dataset, batch_size=2)

    m = AudioVAE(hparams = hparams)

    kld_loss_metric = Metric(name="kld_loss", fn=m.kld_loss)
    recons_loss_metric = Metric(name="recons_loss", fn=m.recons_loss)

    m.train(epochs=100,
            checkpointer=checkpointer,
            datasets={"train": train_dataloader, "val": test_dataloader, "test": test_dataloader},)
            # datasets={"train": train_dataloader, "val": train_dataloader, "test": train_dataloader},)
            # metrics=[kld_loss_metric, recons_loss_metric])
    # m.test(
    #     datasets={"test":test_dataloader}
    # )
    # exit()

    for batch in test_dataloader:
        # print(batch)
        # batch[0] = torch.randn_like(batch[0])
        print(batch[0].shape)
        batch = (batch[0].to('cuda'), batch[1].to('cuda'), batch[2].to('cuda'))
        # print(batch[1])
        # dummy = batch[1][0]
        # batch[1][0] = batch[1][1]
        # batch[1][1] = dummy
        # print(batch[1])

        
        output = m(batch)
        print(len(output[0]))
        # batch_element=7

        for batch_element in range(batch_size):
            original = F.normalize(scale_waveform(output[1][batch_element])).detach().cpu()
            reconstructed = F.normalize(scale_waveform(output[0][batch_element])).detach().cpu()
            print(original)
            print(reconstructed)
            torchaudio.save(f"wavs/{batch_element}_{batch[1][batch_element]}_original.wav", original, sample_rate)
            torchaudio.save(f"wavs/{batch_element}_{batch[1][batch_element]}_reconstructed.wav", reconstructed, sample_rate)

        # for batch_element in range(64):
        #     f, axarr = plt.subplots(2)
        #     f.tight_layout(pad=3.0)
        #     axarr[0].title.set_text("Original")
        #     axarr[0].imshow(output[1][batch_element].squeeze().detach().cpu().numpy().T)

        #     im = axarr[1].title.set_text("Reconstructed")
        #     axarr[1].imshow(output[0][batch_element].squeeze().detach().cpu().numpy().T)
        #     cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        #     f.colorbar(im, cax=cbar_ax)
        #     # f.savefig(f"plots/noise/{batch_element}_{batch[1][batch_element]}.png")
        #     plt.close()
        #     f.savefig(f"plots/reconstructed_latent_space_gaussian/{batch_element}_{batch[1][batch_element]}.png")
        #     # plt.imsave(f"out/{batch_element}.jpg", f)
        break
    
    # m.export("audiovae_onnx_01", "onnx", [23999])
