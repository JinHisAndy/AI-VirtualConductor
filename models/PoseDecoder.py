import torch.nn as nn
import torch
from models.conv_1d import DialtedCNN


class PoseDecoderLSTM(nn.Module):
    """
    Args:
        feature_size (int): input feature dim

    Input: (batch_size, seq_len, feature_size)
    Output: (batch_size, seq_len, 20)

    """

    def __init__(self, input_szie):
        super(PoseDecoderLSTM, self).__init__()

        self.LSTM = torch.nn.LSTM(input_size=input_szie, hidden_size=128, bidirectional=True, num_layers=2,
                                  batch_first=True, dropout=0.5)
        self.out = nn.Sequential(nn.Linear(256, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 20),
                                 nn.Tanh())

    def forward(self, input_feature):
        LSTM_out, _ = self.LSTM(input_feature)
        pose = self.out(LSTM_out)

        return pose


class PoseDecoderTCN(nn.Module):
    """
    Args:
        feature_size (int): input feature dim

    Input: (batch_size, seq_len, feature_size)
    Output: (batch_size, seq_len, 20)

    """

    def __init__(self, input_szie):
        super(PoseDecoderTCN, self).__init__()

        self.TCN = DialtedCNN(input_size=input_szie, output_size=20,n_layers=3,n_channel=64,kernel_size=5,dropout=0.2)

    def forward(self, input_feature):
        out = self.TCN(input_feature)

        return out
