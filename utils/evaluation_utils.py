import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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