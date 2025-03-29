import numpy as np
from typing import Optional, Tuple, Dict

import numpy.typing as npt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
from typing import Optional

import chardet

import mne

from torch.utils.data import Dataset

from EEGNet import EEGNet
from dataset import EEGDataset
device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
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
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            val_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    epoch_loss = val_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
def train_model(epochs,batch_size,model,train_loader,val_loader,patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        # Early Stopping Check
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
    # Load best model state if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model weights restored.")

def test_model(model,test_loader,criterion,device):
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

def get_loaders(epoch: int = 32, batch_size: int = 64, sfreq: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader]:
    base_path = '/Users/cirilla/Documents/Code/ml/eeg/files copy'
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
    sfreq=128
    model = EEGNet(num_classes=3,sfreq=sfreq).to(device)
    epochs = 32
    batch_size = 64
    train_loader, val_loader, test_loader = get_loaders(epoch=epochs, batch_size=batch_size, sfreq=sfreq)

    train_model(epochs,batch_size,model,train_loader,val_loader)
    test_model(model,test_loader,nn.CrossEntropyLoss(),device)
    # Save model
    torch.save(model.state_dict(), 'eegnet.pth')
    # Load model
    # model.load_state_dict(torch.load('eegnet.pth'))
    # model.eval()

if __name__ == '__main__':
    main()