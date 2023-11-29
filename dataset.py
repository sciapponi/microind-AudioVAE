"""
@author: stefanociapponi
"""

import ast, os
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchaudio
import torch

class AudioMNIST(Dataset):

    def __init__(self, data_path, max_len_audio:int = 47998, resample_rate=None):

        
        self.max_len_audio = max_len_audio
        self.data_path = os.path.expanduser(data_path)

        with open(f'{self.data_path}/audioMNIST_meta.txt') as f:
            meta_file=f.read()
        self.meta_dict=ast.literal_eval(meta_file)

        self.wav_file_paths = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    # wav_file_name = file.split('/')[-1]
                    # # extract label and speaker
                    # split_wav_name = wav_file_name.split('_')
                    # label, speaker = split_wav_name[0], split_wav_name[1]
                    # if label in ["0", "1"]:
                    self.wav_file_paths.append(os.path.join(root, file))
        
        #get original sample rate
        if resample_rate is not None:
            waveform, sample_rate = torchaudio.load(self.wav_file_paths[0])
            self.resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
        else:
            self.resampler = None
    
    def resample_all(self, resample_rate):
        #get original sample rate
        waveform, sample_rate = torchaudio.load(self.wav_file_paths[0])
        resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
        for wav_path in self.wav_file_paths:
            waveform, sample_rate = torchaudio.load(wav_path)
            resampled_waveform = resampler(waveform)
            torchaudio.save(wav_path, resampled_waveform, resample_rate)
        
    def __len__(self):
        return len(self.wav_file_paths)

    def normalize(self, waveform):
        min_value = torch.min(waveform)
        max_value = torch.max(waveform)
        normalized_waveform = (waveform - min_value) / (max_value - min_value)
        scaled_waveform = 2 * normalized_waveform - 1

        return scaled_waveform
        # return normalized_waveform


    def __getitem__(self, index)-> any:
        wav_path=self.wav_file_paths[index]
        # loead wav
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
        waveform = F.normalize(waveform)
        sr = sample_rate
        wav_file_name = wav_path.split('/')[-1]
        # extract label and speaker
        split_wav_name = wav_file_name.split('_')
        label, speaker = split_wav_name[0], split_wav_name[1]

        
        waveform = self.resampler(waveform)
        original_length = waveform.shape[1]
        ones_mask = torch.ones(original_length)

        zeros = self.max_len_audio - waveform.shape[1]
        waveform = self.normalize(waveform) #normalize and scale

        waveform = F.pad(waveform, (0, zeros))
        length_mask = F.pad(ones_mask,(0,zeros)).unsqueeze(0)
        # print(length_mask.unsqueeze(0))
        return waveform, int(label), length_mask #, sr, speaker

    def get_speaker_metadata(self, speaker):
        return self.meta_dict[speaker]

if __name__=="__main__":
    base_path = '/home/ste/Code/python/micromind-warmup-task/task_2/AudioMNIST/data'
    
    lengths = []
    srs = []
    
    dataset = AudioMNIST(base_path, resample_rate=22050, max_len_audio=22050)
    dataset[0]
    for waveform, label, original_length in dataset:
        print(original_length)
        # print(waveform.max(), waveform.min())
        lengths.append(waveform.shape[1])
        # srs.append(sr)
    # print(waveform)
    # print(dataset.get_speaker_metadata(speaker))

    # print(set(srs))
    print(f'total {len(set(lengths))}')
    print(f"Max: {max(lengths)}")
    print(f"Min: {min(lengths)}")
    print(f"Max-Min: {max(lengths)-min(lengths)}")