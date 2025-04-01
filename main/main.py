import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import mean_squared_error
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from EEGNet import EEGNet
from dataset import EEGDataset
device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
base_path = '/Users/cirilla/Documents/Code/ml/eeg/files copy'
def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(16, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # MSE
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_mse'], label='Val MSE', color='orange')
    plt.title('Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    # RMSE
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_rmse'], label='Val RMSE', color='red')
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()

    plt.tight_layout()
    plt.show()
def plot_test_metrics(test_metrics: Dict[str, float]):
    labels = list(test_metrics.keys())
    values = list(test_metrics.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['skyblue', 'lightgreen', 'orange', 'salmon'])
    plt.title("Test Set Metrics")
    plt.ylabel("Score")
    plt.ylim(0, max(values) * 1.2)

    # Add value labels on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.3f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    epoch_loss = train_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            val_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    epoch_loss = val_loss / total
    epoch_acc = correct / total
    mse = mean_squared_error(all_targets, all_preds)
    rmse = math.sqrt(mse)

    return epoch_loss, epoch_acc, mse, rmse
def train_model(epochs, batch_size, model, train_loader, val_loader, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_mse': [],
        'val_rmse': []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_mse, val_rmse = evaluate(model, val_loader, criterion, device)

        # Log values
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mse'].append(val_mse)
        history['val_rmse'].append(val_rmse)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | MSE: {val_mse:.4f} | RMSE: {val_rmse:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement in val loss. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                # break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model weights restored.")

    return history

def test_model(model, test_loader, criterion, device):
    test_loss, test_acc, test_mse, test_rmse = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f}")

    return {
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Test MSE': test_mse,
        'Test RMSE': test_rmse
    }

def get_loaders(epoch: int = 32, batch_size: int = 64, sfreq: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dataset = EEGDataset(data_dir=base_path, sfreq=sfreq, duration=1.0)

    # Step 1: Create indices for the full dataset
    total_size = len(dataset)
    indices = np.arange(total_size)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])  # if label is at index 1
    # Step 2: Split into Train and Temp (temp will be split into val and test)
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    # Step 3: Split Temp into Validation and Test
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx])

    # Step 4: Create subsets
    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)
    test_data = Subset(dataset, test_idx)

    # Step 5: Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    sfreq = 128
    model = EEGNet(num_classes=3, sfreq=sfreq).to(device)
    epochs = 32
    batch_size = 64
    train_loader, val_loader, test_loader = get_loaders(epoch=epochs, batch_size=batch_size, sfreq=sfreq)

    history = train_model(epochs, batch_size, model, train_loader, val_loader)
    plot_metrics(history)  # 👈 add this line

    torch.save(model.state_dict(), 'eegnet.pth')
    # model.load_state_dict(torch.load('eegnet.pth'))
    test_metrics = test_model(model, test_loader, nn.CrossEntropyLoss(), device)
    plot_test_metrics(test_metrics)

if __name__ == '__main__':
    m={
        'Test Loss': 0.8410,
        'Test Accuracy': 0.7015,
        'Test MSE': 0.6078,
        'Test RMSE': 0.7786
    }
    plot_test_metrics(m)