from typing import Dict
from sklearn.metrics import mean_absolute_error
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
    plt.title('Validation Loss')
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

    # MAE
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['val_mae'], label='Val MAE', color='orange')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # RMAE
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['val_rmae'], label='Val RMAE', color='red')
    plt.title('Validation RMAE')
    plt.xlabel('Epoch')
    plt.ylabel('RMAE')
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
    mae = mean_absolute_error(all_targets, all_preds)
    rmae = math.sqrt(mae)

    return epoch_loss, epoch_acc, mae, rmae
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
        'val_mae': [],
        'val_rmae': []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_mae, val_rmae = evaluate(model, val_loader, criterion, device)

        # Log values
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mae'].append(val_mae)
        history['val_rmae'].append(val_rmae)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | MAE: {val_mae:.4f} | RMAE: {val_rmae:.4f}")

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
    test_loss, test_acc, test_mae, test_rmae = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"Test MAE: {test_mae:.4f} | Test RMAE: {test_rmae:.4f}")

    return {
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Test MAE': test_mae,
        'Test RMAE': test_rmae
    }
