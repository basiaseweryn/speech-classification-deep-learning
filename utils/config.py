import os

seed = 42

ALL_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown"]
SUBSET_CLASSES = ["yes", "no", "up", "down", "left", "right"]

audio_params = {
    "sample_rate": 16000,
    "duration": 1.0,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 512,
}

data_dir = "/kaggle/working/train/audio"
trained_models_dir = "./trained_models"

COMMON_BATCH_SIZE = 32
COMMON_LR = 0.0005
COMMON_SCHEDULER = "ReduceLROnPlateau"
COMMON_EPOCHS = 25

experiments = {
    "stage_1_baseline_cnn": {
        "model_type": "BaselineCNN",
        "model_params": {
            "n_filters": 32, 
            "drop_rate": 0.0
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": True
    },

    "stage_1_baseline_transformer_scratch": {
        "model_type": "TransformerScratch",
        "model_params": {
            "n_layers": 4,
            "n_heads": 8, 
            "drop_rate": 0.0,
            "patch_size": 4
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": True
    },

    "stage_1_baseline_transformer_pretrained": {
        "model_type": "PretrainedTransformer",
        "model_params": {
            "strategy": "none"
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": True
    }
}