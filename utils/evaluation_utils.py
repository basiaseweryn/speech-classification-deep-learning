import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from config import ALL_CLASSES
from data_utils import get_dataloaders
from models import get_model

def plot_confusion_matrix(targets, preds, classes, save_path):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('confusion matrix')
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curves(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # plot loss (to monitor learning dynamics and overfitting)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='train loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='validation loss', marker='o')
    plt.title('learning dynamics: loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], label='train acc', marker='o')
    plt.plot(epochs, history['val_acc'], label='val acc', marker='o')
    plt.title('learning dynamics: accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()

def evaluate_cascade_fast(model_gk, model_spec, val_loader, device):
    model_gk.eval()
    model_spec.eval()
    all_preds, all_targets = [], []
    
    GK_SILENCE, GK_UNKNOWN, GK_COMMAND = 0, 1, 2
    
    spec_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    spec_mapping = {i: ALL_CLASSES.index(name) for i, name in enumerate(spec_labels)}
    idx_silence, idx_unknown = ALL_CLASSES.index("silence"), ALL_CLASSES.index("unknown")

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Ewaluacja kaskady"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            out_gk = model_gk(inputs)
            _, pred_gk = out_gk.max(1)
            
            final_batch_preds = torch.zeros_like(pred_gk)
            final_batch_preds[pred_gk == GK_SILENCE] = idx_silence
            final_batch_preds[pred_gk == GK_UNKNOWN] = idx_unknown
            
            cmd_mask = (pred_gk == GK_COMMAND)
            if cmd_mask.any():
                out_spec = model_spec(inputs[cmd_mask])
                _, pred_spec = out_spec.max(1)
                mapped_spec = torch.tensor([spec_mapping[p.item()] for p in pred_spec], device=device)
                final_batch_preds[cmd_mask] = mapped_spec
            
            all_preds.extend(final_batch_preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return all_targets, all_preds