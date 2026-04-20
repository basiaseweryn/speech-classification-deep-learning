import os

# global seed for reproducibility
seed = 42

# working mode flag
use_subset = True

# classes definition
all_classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown"]
subset_classes = ["yes", "no", "up", "down"]

active_classes = subset_classes if use_subset else all_classes
num_classes = len(active_classes)

# audio parameters
audio_params = {
    "sample_rate": 16000,
    "duration": 1.0,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 512,
}

# paths
# data_dir = "/kaggle/input/tensorflow-speech-recognition-challenge/train/audio"
data_dir = "/kaggle/working/train/train/audio"
trained_models_dir = "./trained_models"

# experiment configurations with dynamic parameters
experiments = {
    "stage_1_baseline_cnn_default": {
        "model_type": "BaselineCNN",
        "model_params": {"n_filters": 32, "drop_rate": 0.3},
        "epochs": 15,
        "batch_size": 64,
        "lr": 0.001,
        "scheduler": "ReduceLROnPlateau"
    },
    "stage_2_cnn_investigation_filters": {
        "model_type": "BaselineCNN",
        "model_params": {"n_filters": 64, "drop_rate": 0.3},
        "epochs": 15,
        "batch_size": 64,
        "lr": 0.001,
        "scheduler": "StepLR"
    },
    "stage_2_transformer_scratch_default": {
        "model_type": "TransformerScratch",
        "model_params": {"n_layers": 4, "n_heads": 8, "drop_rate": 0.1},
        "epochs": 20,
        "batch_size": 32,
        "lr": 0.0005,
        "scheduler": "CosineAnnealingLR"
    },
    "stage_2_transformer_pretrained_freeze": {
        "model_type": "PretrainedTransformer",
        "model_params": {"strategy": "freeze"},
        "epochs": 15,
        "batch_size": 32,
        "lr": 0.0001,
        "scheduler": "ReduceLROnPlateau"
    }
}