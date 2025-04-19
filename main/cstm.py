import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Reshape to (batch_size, seq_len, input_size) for LSTM
        x = x.squeeze(1)  # From (batch_size, 1, channels, samples) to (batch_size, samples, channels)
        out = self.cnn(x)
        # LSTM expects (batch_size, seq_len, input_size)
        out = out.permute(0, 2, 1)  # Change to (batch_size, seq_len, input_size)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out
