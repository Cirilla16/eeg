import numpy as np
from typing import Optional, Tuple, Dict

import numpy.typing as npt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split

from main.EEGNet import EEGNet
from main.dataset import EEGDataset


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


def load_eeg_data_nd_array(
    edf_path: str,
    event_labels: Optional[list[str]] = None,
    event_window: Tuple[float, float] = (0, 1),
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None
) -> Tuple[
    npt.NDArray[np.float32],      # data: shape (30, 64, 161)
    npt.NDArray[np.int32],        # labels: shape (30,)
    float,                        # sampling frequency
    Dict[str, int]                # event label mapping
]:
    # Load file
    raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel='auto', verbose=False)
    raw.pick_types(eeg=True)

    # Get sampling rate
    sfreq = raw.info['sfreq']

    # Extract events and map labels
    events, event_id = mne.events_from_annotations(raw)

    if event_labels is None:
        event_labels = list(event_id.keys())

    selected_event_id = {label: event_id[label] for label in event_labels}

    # Epoching: extract segments around events
    epochs = mne.Epochs(
        raw, events, event_id=selected_event_id,
        tmin=event_window[0], tmax=event_window[1],
        baseline=baseline, preload=True, verbose=False
    )

    # Get data and labels
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]  # shape: (n_epochs,)

    # Map label integers to 0-based class indices
    label_to_index = {eid: idx for idx, eid in enumerate(selected_event_id.values())}
    labels = np.vectorize(label_to_index.get)(labels)
    return data, labels, sfreq, selected_event_id
def load_eeg_data_tensor_EEGNet(edf_path, event_labels=None, event_window=(0, 1), baseline=None):
    data, labels,sfreq,selected_event_id=load_eeg_data_nd_array(edf_path, event_labels, event_window, baseline)

    # Convert to torch tensors
    X = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # [N(batch_size), 1, C(channels), T(time)]
    y = torch.tensor(labels, dtype=torch.long)  # [N]

    return X, y, sfreq, selected_event_id

def load_eeg_from_records(base_dir):

    data_dict = {}

    with open(f'{base_dir}/RECORDS', 'r') as f:
        edf_paths = f.read().splitlines()

    for rel_path in edf_paths:
        full_path = os.path.join(base_dir, rel_path)

        if not os.path.isfile(full_path):
            print(f"File not found: {full_path}")
            continue

        try:
            load_eeg_data_tensor_EEGNet(full_path)
            raw = mne.io.read_raw_edf(full_path, preload=True, verbose=False)
            data_dict[rel_path] = raw
            print(f"Loaded: {rel_path}")

        except Exception as e:
            print(f"Failed to load {rel_path}: {e}")

    return data_dict
def extract():
    base_path='/Users/cirilla/Documents/Code/ml/eeg/files'
    dataset = EEGDataset(data_dir=base_path, sfreq=256, duration=1.0)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(len(dataset))

    X, y = dataset[0]
    print("Shape:", X.shape)  # (1, channels, timepoints)
    print("Label:", y)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    model = EEGNet(in_channels=64, num_classes=4, input_samples=128).to(device)
    # output = model(torch.randn(8, 64, 128))  # batch_size=8
    # print(output.shape)  # [8, 4]
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    base_path = '/Users/cirilla/Documents/Code/ml/eeg/files'
    dataset = EEGDataset(data_dir=base_path, sfreq=256, duration=1.0)
    epochs = 32
    batch_size = 64
    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(0), labels.to(0)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss, val_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(0), labels.to(0)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Acc: {train_correct / len(train_data):.4f} | Val Acc: {val_correct / len(val_data):.4f}")

if __name__ == '__main__':
    main()