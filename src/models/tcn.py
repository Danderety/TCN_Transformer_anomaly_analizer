import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = int(chomp_size)
    def forward(self, x): return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__(); padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding); self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    def forward(self, x):
        r = x
        y = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.relu2(self.chomp2(self.conv2(y))))
        if self.downsample is not None: r = self.downsample(r)
        return F.relu(y + r)

class TCNEncoder(nn.Module):
    def __init__(self, d_model, num_layers=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(*[TCNBlock(d_model, d_model, kernel_size, 2**i, dropout) for i in range(num_layers)])
    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)
