import torch
import torch.nn as nn
import timm 
from config import num_classes

# 1. baseline cnn
class BaselineCNN(nn.Module):
    def __init__(self, n_filters=32, drop_rate=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(n_filters, n_filters*2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(n_filters*2, n_filters*4, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(n_filters*4, num_classes))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

# 2. transformer non-pretrained
def get_transformer_scratch(n_layers=4, n_heads=8, drop_rate=0.1):
    # dynamic params passed directly to the timm builder
    model = timm.create_model(
        'vit_tiny_patch16_224', 
        pretrained=False,
        in_chans=1,
        num_classes=num_classes,
        drop_rate=drop_rate,
        depth=n_layers,
        num_heads=n_heads,
        img_size=(64, 32)
    )
    return model

# 3. pretrained transformer
def get_pretrained_transformer(strategy="freeze"):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, in_chans=1, num_classes=num_classes)
    
    # fine-tuning strategy implementation
    if strategy == "freeze":
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze classification head only
        for param in model.head.parameters():
            param.requires_grad = True
            
    # if strategy is anything else, leaves layers unfrozen
    return model

def get_model(config):
    model_type = config.get("model_type")
    params = config.get("model_params", {})
    
    if model_type == "BaselineCNN":
        return BaselineCNN(**params)
    elif model_type == "TransformerScratch":
        return get_transformer_scratch(**params)
    elif model_type == "PretrainedTransformer":
        return get_pretrained_transformer(**params)
    else:
        raise ValueError(f"unknown model type: {model_type}")