import os
from typing import Optional
import torch
import numpy as np
import mne
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data_dir, sfreq=128, duration=4.0, transform=None,fraction=1.0, seed=42):
        self.data_dir = data_dir
        self.sfreq = sfreq
        self.duration = duration
        self.transform = transform
        self.epochs = []
        self.labels = []
        self._load_all_data()

    def _load_all_data(self):
        for subject in sorted(os.listdir(self.data_dir)):
            subj_path = os.path.join(self.data_dir, subject)
            if os.path.isdir(subj_path):
                edf_files = [f for f in os.listdir(subj_path) if f.endswith(".edf")]
                for edf_file in edf_files:
                    edf_path = os.path.join(subj_path, edf_file)
                    self._process_edf_file(edf_path)

    def _process_edf_file(self, edf_path):
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.resample(self.sfreq)
        data = raw.get_data()  # shape: (n_channels, n_samples)

        annotations = raw.annotations
        # Extract events and map labels
        event_labels: Optional[list[str]] = None
        events, event_id = mne.events_from_annotations(raw)
        if event_labels is None:
            event_labels = list(event_id.keys())
            print(event_labels)

        selected_event_id = {label: event_id[label] for label in event_labels}
        for desc, onset, dur in zip(annotations.description, annotations.onset, annotations.duration):
            if desc not in selected_event_id:
                continue

            label = selected_event_id[desc]
            start_sample = int(onset * self.sfreq)
            end_sample = start_sample + int(self.duration * self.sfreq)

            if end_sample <= data.shape[1]:
                epoch = data[:, start_sample:end_sample]
                self.epochs.append(epoch.astype(np.float32))
                self.labels.append(label)
        return
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        X = self.epochs[idx]
        y = self.labels[idx]

        # EEGNet expects shape: (1, channels, time)
        X = np.expand_dims(X, axis=0)
        if self.transform:
            X = self.transform(X)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)