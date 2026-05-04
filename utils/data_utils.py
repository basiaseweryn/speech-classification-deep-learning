import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from config import ALL_CLASSES, SUBSET_CLASSES, audio_params, data_dir

class SpeechDataset(Dataset):
    def __init__(self, data_dir, classes, config=audio_params, subset_mode=False, task_type="standard", file_list=None):
        self.data_dir = data_dir
        self.classes = classes
        self.config = config
        self.subset_mode = subset_mode
        self.task_type = task_type
        self.file_list = set(file_list) if file_list is not None else None
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
            if not os.path.isdir(folder_path): continue
            
            if self.task_type == "unknown_filter":
                if folder == "_background_noise_":
                    self._load_files(folder_path, "silence", is_bg=True, multiply=1)
                elif folder in core_words:
                    self._load_files(folder_path, "command", is_bg=False)
                else:
                    self._load_files(folder_path, "unknown", is_bg=False)
                    
            elif self.task_type == "command_specialist":
                if folder in self.classes:
                    self._load_files(folder_path, folder, is_bg=False)
                    
            else: 
                if self.subset_mode:
                    if folder in self.classes:
                        self._load_files(folder_path, folder, is_bg=False)
                else:
                    if folder == "_background_noise_":
                        if "silence" in self.classes:
                            self._load_files(folder_path, "silence", is_bg=True, multiply=400)
                    elif folder in core_words:
                        self._load_files(folder_path, folder, is_bg=False)
                    else:
                        if "unknown" in self.classes:
                            self._load_files(folder_path, "unknown", is_bg=False)

    def _load_files(self, folder_path, label, is_bg=False, multiply=1):
        if label not in self.label_to_idx: return
        label_idx = self.label_to_idx[label]
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                if self.file_list is not None and file_path not in self.file_list:
                    continue
                for _ in range(multiply):
                    self.samples.append((file_path, label_idx, is_bg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label_idx, is_bg = self.samples[idx]
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if is_bg:
            if waveform.shape[1] > self.target_samples:
                start = random.randint(0, waveform.shape[1] - self.target_samples)
                waveform = waveform[:, start:start + self.target_samples]
            else:
                waveform = F.pad(waveform, (0, self.target_samples - waveform.shape[1]))
        else:
            if waveform.shape[1] < self.target_samples:
                waveform = F.pad(waveform, (0, self.target_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.target_samples]
            
        spec = self.to_db(self.transform(waveform))
        return spec, label_idx

def get_dataloaders(exp_config, num_workers=4):
    reduced = exp_config.get("reduced_classes", False)
    task_type = exp_config.get("task_type", "standard")
    
    if task_type == "unknown_filter":
        classes = ["silence", "unknown", "command"]
    elif task_type == "command_specialist":
        classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    else:
        classes = SUBSET_CLASSES if reduced else ALL_CLASSES
    
    all_wavs = []
    all_labels = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path): continue
        for f in os.listdir(folder_path):
            if f.endswith(".wav"):
                all_wavs.append(os.path.join(folder_path, f))
                all_labels.append(folder)
    
    train_files, val_files = train_test_split(
        all_wavs, test_size=0.2, random_state=42, shuffle=True, stratify=all_labels
    )
    
    train_ds = SpeechDataset(data_dir, classes, subset_mode=reduced, task_type=task_type, file_list=train_files)
    val_ds = SpeechDataset(data_dir, classes, subset_mode=reduced, task_type=task_type, file_list=val_files)
    
    use_pin_memory = torch.cuda.is_available()
    sampler = None
    shuffle_train = True
    
    if exp_config.get("sampling") == "weighted":
        train_targets = [s[1] for s in train_ds.samples]
        class_counts = torch.bincount(torch.tensor(train_targets))
        class_weights = 1.0 / class_counts.float()
        sample_weights = torch.tensor([class_weights[t] for t in train_targets])
        
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        shuffle_train = False
    
    train_loader = DataLoader(train_ds, batch_size=exp_config["batch_size"], shuffle=shuffle_train, sampler=sampler, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=exp_config["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    
    return train_loader, val_loader