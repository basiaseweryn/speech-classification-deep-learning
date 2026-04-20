import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import active_classes, audio_params, data_dir, use_subset

class SpeechDataset(Dataset):
    def __init__(self, data_dir, classes, config=audio_params, subset_mode=use_subset):
        self.data_dir = data_dir
        self.classes = classes
        self.config = config
        self.subset_mode = subset_mode
        self.samples = []
        
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.target_samples = int(config["sample_rate"] * config["duration"])
        
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["sample_rate"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"]
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self._prepare_data()

    def _prepare_data(self):
        core_words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        
        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            if self.subset_mode:
                # load only specific classes, ignore everything else
                if folder in self.classes:
                    self._load_files(folder_path, folder, is_bg=False)
            else:
                # handle full dataset mode
                if folder == "_background_noise_":
                    if "silence" in self.classes:
                        self._load_files(folder_path, "silence", is_bg=True, multiply=400)
                elif folder in core_words:
                    self._load_files(folder_path, folder, is_bg=False)
                else:
                    # all other folders map to unknown
                    if "unknown" in self.classes:
                        self._load_files(folder_path, "unknown", is_bg=False)

    def _load_files(self, folder_path, label, is_bg=False, multiply=1):
        if label not in self.label_to_idx:
            return
            
        label_idx = self.label_to_idx[label]
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                # include same file multiple times to balance the dataset
                for _ in range(multiply):
                    self.samples.append((file_path, label_idx, is_bg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_idx, is_bg = self.samples[idx]
        waveform, sr = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if is_bg:
            # background noise: take random 1 second segment, pad if shorter
            if waveform.shape[1] > self.target_samples:
                start = random.randint(0, waveform.shape[1] - self.target_samples)
                waveform = waveform[:, start:start + self.target_samples]
            else:
                waveform = F.pad(waveform, (0, self.target_samples - waveform.shape[1]))
        else:
            # normal speech: take first 1 second, pad if shorter
            if waveform.shape[1] < self.target_samples:
                waveform = F.pad(waveform, (0, self.target_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.target_samples]
            
        # generate spectrogram
        spec = self.to_db(self.transform(waveform))
        return spec, label_idx

def get_dataloaders(config, num_workers=4):
    full_ds = SpeechDataset(data_dir, active_classes)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    
    # keeping the split reproducible
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader