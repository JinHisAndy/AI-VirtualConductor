import torch
import torch.nn as nn


class PoseEncoderNoPool(nn.Module):
    """convolution network on pose sequence, convert pose sequence to latent embeddings
    Args:
        feature_size (int): dim of input feature type
        hidden_size (int): output embedding size
    Input: (batch_size, seq_len, input_size)
    Output: (batch_size seq_len, hidden_size,)
    """

    def __init__(self):
        super(PoseEncoderNoPool, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=20, out_channels=80, kernel_size=25, stride=1, padding=12,groups=20)
        self.ln1 = nn.LayerNorm([80])
        self.conv2 = nn.Conv1d(in_channels=80, out_channels=32, kernel_size=25, stride=1, padding=12)
        self.ln2 = nn.LayerNorm([32])

    def forward(self, y):
        h1 = self.conv1(y.transpose(1, 2)).transpose(1, 2)
        h1 = torch.relu(self.ln1(h1))
        h1 = torch.avg_pool1d(h1,kernel_size=3,stride=1,padding=1)

        h2 = self.conv2(h1.transpose(1, 2)).transpose(1, 2)
        h2 = torch.sigmoid(self.ln2(h2))

        return torch.cat([h1,h2],dim=2), h2


class PoseEncoderDr(nn.Module):
    """convolution network on pose sequence, convert pose sequence to latent embeddings
    Args:
        feature_size (int): dim of input feature type
        hidden_size (int): output embedding size
    Input: (batch_size, seq_len, input_size)
    Output: (batch_size seq_len, hidden_size,)
    """

    def __init__(self):
        super(PoseEncoderDr, self).__init__()

        self.conv1_group= nn.Conv1d(in_channels=20, out_channels=80, kernel_size=15, stride=1, padding=7,groups=20)
        self.conv1_nogroup = nn.Conv1d(in_channels=20, out_channels=80, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(in_channels=160, out_channels=128, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=15, stride=1, padding=7)


    def forward(self, y):
        y = y.transpose(1, 2)
        h1_group = self.conv1_group(y)
        h1_nogroup = self.conv1_nogroup(y)
        h1 = torch.cat([h1_group,h1_nogroup],dim=1)
        h1 = torch.relu(h1)
        h1 = torch.max_pool1d(h1, kernel_size=5, stride=5)

        h2 = self.conv2(h1)
        h2 = torch.relu(h2)
        h2 = torch.max_pool1d(h2, kernel_size=5, stride=5)

        h3 = self.conv3(h2)
        h3 = torch.relu(h3)
        h3 = torch.max_pool1d(h3, kernel_size=5, stride=5)

        return h3.transpose(1, 2), h1.transpose(1, 2)


class PoseEncoderStyle(nn.Module):
    """convolution network on pose sequence, convert pose sequence to latent embeddings
    Args:
        feature_size (int): dim of input feature type
        hidden_size (int): output embedding size
    Input: (batch_size, seq_len, input_size)
    Output: (batch_size seq_len, hidden_size,)
    """

    def __init__(self):
        super(PoseEncoderStyle, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=20, out_channels=32, kernel_size=101, stride=1, padding=50)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=25, stride=1, padding=12)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=25, stride=1, padding=12)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=25, stride=1, padding=12)

    def forward(self, y):
        h1 = self.conv1(y.transpose(1, 2))
        h1 = torch.relu(h1)
        h1 = torch.max_pool1d(h1, kernel_size=5, stride=5)

        h2 = self.conv2(h1)
        h2 = torch.relu(h2)
        h2 = torch.max_pool1d(h2, kernel_size=5, stride=5)

        h3 = self.conv3(h2)
        h3 = torch.relu(h3)
        h3 = torch.max_pool1d(h3, kernel_size=5, stride=5)

        h4 = self.conv3(h3)
        h4 = torch.relu(h4)
        h4 = torch.max_pool1d(h4, kernel_size=5, stride=5)

        return h4.transpose(1, 2)


if __name__ == '__main__':
    pass
