from torch.nn.utils import weight_norm
import torch
from torch import nn


class Conv2dSpectrum(nn.Module):
    """
    convolution network on spectrum features, convert spectrums to latent embeddings
    Args:
        feature_size (int): dim of input feature type
        hidden_size (int): output embedding size
    Input: (batch_size, feature_size, seq_len)
    Output: (batch_size, hidden_size, seq_len)
    """
    def __init__(self, feature_size, hidden_size):
        super(Conv2dSpectrum, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size * 16, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, x):
        x = self.convolutions(torch.unsqueeze(x, 1))
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=2)
        x = self.fc(x)
        return x

