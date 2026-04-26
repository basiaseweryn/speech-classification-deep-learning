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
    },

# 30 runs
"stage_2_cnn_deep_search": {
        "model_type": "BaselineCNN",
        "model_params": {
            "n_filters": [16, 32, 64, 128, 256, 512],
            "drop_rate": [0.0, 0.1, 0.2, 0.3, 0.5]
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": True                        
    },

    # 45 runs
    "stage_2_transformer_deep_search": {
        "model_type": "TransformerScratch",
        "model_params": {
            "n_layers": [2, 4, 8],
            "n_heads": [4, 8, 16],                 
            "drop_rate": [0.0, 0.1, 0.2, 0.3, 0.5],          
            "patch_size": 4                             
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": True
    },

    # 3 runs
    "stage_2_pretrained_strategy_search": {
        "model_type": "PretrainedTransformer",
        "model_params": {
            "strategy": ["freeze", "partial", "none"]   # "freeze, partial, etc." 
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": True
    },
    # STAGE 4 - training on full dataset
     "stage_4_pretrained": {
        "model_type": "PretrainedTransformer",
        "model_params": {
            "strategy": ["partial", "none"]
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": False
    },
    "stage_4_cnn_1": {. 				# best F1=0.966, small overfitting gap (0.24%), convergence in 6th epoch. main candidate
        "model_type": "BaselineCNN",
        "model_params": {
            "n_filters": 128,
            "drop_rate": 0.3 
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": False                        
    },
    "stage_4_cnn_2": {					# big network, no regularization
        "model_type": "BaselineCNN",
        "model_params": {
            "n_filters": 512,
            "drop_rate": 0.0 
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": False                        
    },
    "stage_4_cnn_3": { 				# small network (but not the smallest possible), big regularization  
        "model_type": "BaselineCNN",
        "model_params": {
            "n_filters": 32,
            "drop_rate": 0.5 
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": False                        
    },

    "stage_4_transformer_1": { 					# best F1, main candidate
        "model_type": "TransformerScratch",
        "model_params": {
            "n_layers": 4,
            "n_heads": 4,
            "drop_rate”: 0.3,
            "patch_size": 4                             
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": False
    },
    "stage_4_transformer_2": { 					# one of the highest F1 score, but a bigger network 
        "model_type": "TransformerScratch",
        "model_params": {
            "n_layers": 8,
            "n_heads": 4,
            "drop_rate”: 0.2,
            "patch_size": 4                             
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": False
    },
    "stage_4_transformer_3": { 					# parameters of the main candidate but without regularization
        "model_type": "TransformerScratch",
        "model_params": {
            "n_layers": 4,
            "n_heads": 4,
            "drop_rate”: 0.0,
            "patch_size": 4                             
        },
        "epochs": COMMON_EPOCHS,
        "batch_size": COMMON_BATCH_SIZE,
        "lr": COMMON_LR,
        "scheduler": COMMON_SCHEDULER,
        "reduced_classes": False
    },
}
