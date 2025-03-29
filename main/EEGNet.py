import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchgen.executorch.api.et_cpp import return_names

device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")

class EEGNet(nn.Module):
    def __init__(self, num_classes=64,sfreq=128):
        super().__init__()

        self.num_class = num_classes
        self.sfreq=sfreq
        self.channel = 64  # #number of electrode
        # self.signal_length = 256      #sfreq: number of sample in each tarial
        self.F1 = 8  # #number of temporal filters
        self.D = 2  # depth multiplier (number of spatial filters)
        self.F2 = self.D * self.F1  # number of pointwise filters

        # layer 1: temporal convolution
        # 64: kernel_length
        conv2d = nn.Conv2d(1, self.F1, (1,64), padding=(0, 64 // 2), bias=False)
        Batch_normalization_1 = nn.BatchNorm2d(self.F1)
        self.firstconv = nn.Sequential(
            conv2d,
            Batch_normalization_1,
        )
        # layer 2: spatial convolution
        Depthwise_conv2D = nn.Conv2d(self.F1, self.D * self.F1, (self.channel, 1), groups=self.F1, bias=False)
        Batch_normalization_2 = nn.BatchNorm2d(self.D * self.F1)
        Average_pooling2D_1 = nn.AvgPool2d((1,4))
        Dropout = nn.Dropout2d(0.2)
        self.depthwiseConv = nn.Sequential(
            Depthwise_conv2D,
            Batch_normalization_2,
            nn.ELU(),
            Average_pooling2D_1,
            nn.Dropout2d(0.2),
        )

        # layer 3: Separable Convolution
        Separable_conv2D_depth = nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, 16), padding=(0,8), groups=self.D * self.F1, bias=False)
        Separable_conv2D_point = nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False)
        Batch_normalization_3 = nn.BatchNorm2d(self.F2)
        Average_pooling2D_2 = nn.AvgPool2d((1,8))
        self.separableConv = nn.Sequential(
            Separable_conv2D_depth,
            Separable_conv2D_point,
            Batch_normalization_3,
            nn.ELU(),
            Average_pooling2D_2,
            nn.Dropout(p=0.25)
        )
        # layer 4
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(self.F2 * 1 * (sfreq // (4 * 8)), self.num_class) # self.F2 * round(self.signal_length / 32)
        self.Softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # layer 1
        y = self.firstconv(x)
        # layer 2
        y = self.depthwiseConv(y)
        # layer 3
        y = self.separableConv(y)
        # layer 4
        y = self.Flatten(y)
        y = self.Dense(y)
        y = self.Softmax(y)
        return y


class EEGNet0(nn.Module):
    def __init__(self, in_channels, num_classes, input_samples):
        super(EEGNet, self).__init__()

        # First temporal convolution
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )

        # Depthwise convolution
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(in_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.25)
        )

        # Separable convolution
        self.separableConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.25)
        )

        # Calculate the output size after conv layers to define final dense layer
        self._output_size = self._compute_output_size(in_channels, input_samples)

        # Final classifier
        self.classify = nn.Linear( 128,64)

    def _compute_output_size(self, in_channels, input_samples):
        # Dummy input to determine final output shape
        with torch.no_grad():
            x = torch.zeros(1, 1, in_channels, input_samples)
            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        # Input shape: (batch, channels, time)
        # x = x.unsqueeze(1)  # reshape to (batch, 1, channels, time)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        x = self.classify(x)
        return x


def try_random():
    # Example input
    batch_size = 16
    channels = 64
    samples = 128
    num_classes = 3

    # Model instantiation
    model = EEGNet(num_classes=num_classes).to(device)
    # Dummy input: (batch_size, 1, channels, samples)
    x = torch.randn(batch_size, 1, channels, samples)

    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)  # Expected: (16, num_classes)
def try_backup():
    signal = torch.randn(1280, 1, 22, 200)
    label = torch.randint(2, (1280,))
    train_batch_size = 256

    dataset = TensorDataset(signal, label)

    data_loader = DataLoader(dataset,
                             batch_size=train_batch_size,
                             shuffle=True)

    print("train batch size:", data_loader.batch_size,
          ", num of batch:", len(data_loader))

    model = EEGNet().to(device)

    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)  # (256,1,22,200)
        targets = targets.to(device)  # (256,)

        outputs = model(inputs)
        print(outputs)  # (256,5)
        loss = nn.CrossEntropyLoss()(outputs, targets)
if __name__ == '__main__':
    try_random()

