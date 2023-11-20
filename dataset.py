"""
@author: stefanociapponi
"""

import ast, os
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchaudio
import torch

class AudioMNIST(Dataset):

    def __init__(self, data_path, max_len_audio:int = 47998):

        self.max_len_audio = max_len_audio
        self.data_path = os.path.expanduser(data_path)

        with open(f'{self.data_path}/audioMNIST_meta.txt') as f:
            meta_file=f.read()
        self.meta_dict=ast.literal_eval(meta_file)

        self.wav_file_paths = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    self.wav_file_paths.append(os.path.join(root, file))
        
    # to implement -> should be done before reading files
    def remove_long_samples(max_length):
        pass

    def __len__(self):
        return len(self.wav_file_paths)

    def __getitem__(self, index)-> any:
        wav_path=self.wav_file_paths[index]
        # loead wav
        waveform, sample_rate = torchaudio.load(wav_path)
        sr = sample_rate
        wav_file_name = wav_path.split('/')[-1]
        # extract label and speaker
        split_wav_name = wav_file_name.split('_')
        label, speaker = split_wav_name[0], split_wav_name[1]

        zeros = self.max_len_audio - waveform.shape[1]
        waveform = F.pad(waveform, (0, zeros))
        return waveform, int(label), sr, speaker

    def get_speaker_metadata(self, speaker):
        return self.meta_dict[speaker]

if __name__=="__main__":
    base_path = '/home/ste/Code/python/micromind-warmup-task/task_2/AudioMNIST/data'
    
    lengths = []
    srs = []
    
    dataset = AudioMNIST(base_path, 16000)
    dataset[0]
    for waveform, sr, label, speaker in dataset:
        lengths.append(waveform.shape[1])
        srs.append(sr)
    # print(waveform)
    # print(dataset.get_speaker_metadata(speaker))

    print(set(srs))
    print(f'total {len(set(lengths))}')
    print(f"Max: {max(lengths)}")
    print(f"Min: {min(lengths)}")
    print(f"Max-Min: {max(lengths)-min(lengths)}")