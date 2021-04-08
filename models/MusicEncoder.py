import torch
from torch import nn

feature_dict = {
    'spectral_centroid': [0, 1],
    'spectral_bandwidth': [1, 2],
    'onset_envelope': [3, 4],
    'pulse': [3, 4],
    'pulse_lognorm': [4, 5],
    'dtempo': [5, 6],
    'tempogram': [6, 390],
    'mfcc': [390, 410],
    'mel_spectrogram': [410, 538],
    'cqt': [538, 622]
}


class FeatureEncoder1D(nn.Module):
    """
    Encoding 1D MIR features with grouped 1d convolutions

    Input: (batch_size, seq_len, in_features)
    Output:(batch_size, seq_len, in_features * K)
    """

    def __init__(self, in_features, K):
        super(FeatureEncoder1D, self).__init__()

        self.conv_1 = nn.Conv1d(in_features, K * in_features, kernel_size=3, padding=1)
        self.BN_1 = nn.BatchNorm1d(K * in_features)

        self.conv_2 = nn.Conv1d(K * in_features, K * in_features, kernel_size=3, padding=1)
        self.BN_2 = nn.BatchNorm1d(K * in_features)

        self.conv_3 = nn.Conv1d(K * in_features, K * in_features, kernel_size=3, padding=1)
        self.BN_3 = nn.BatchNorm1d(K * in_features)

        self.conv_4 = nn.Conv1d(K * in_features, K * in_features, kernel_size=3, padding=1)
        self.BN_4 = nn.BatchNorm1d(K * in_features)

        self.conv_5 = nn.Conv1d(K * in_features, K * in_features, kernel_size=15, padding=7)
        self.BN_5 = nn.BatchNorm1d(K * in_features)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.BN_1(self.conv_1(x)))
        x = torch.relu(self.BN_2(self.conv_2(x)))
        x = torch.relu(self.BN_3(self.conv_3(x)))
        x = torch.relu(self.BN_4(self.conv_4(x)))
        x = torch.sigmoid(self.BN_5(self.conv_5(x)))
        x = x.transpose(1, 2)

        return x


class FeatureEncoder2D(nn.Module):
    """
    Encoding 2D MIR features （spectrum features） with 2d convolutions

    Input: (batch_size, seq_len, in_features)
    Output:(batch_size, seq_len, out_features)
    """

    def __init__(self, in_features, out_features):
        super(FeatureEncoder2D, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(in_features * 8, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.Sigmoid())

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        x = torch.flatten(x, start_dim=2)
        x = self.fc(x)

        return x


class MusicEncoderPool(nn.Module):
    def __init__(self):
        super(MusicEncoderPool, self).__init__()

        self.normalize_6 = nn.Sequential(nn.LayerNorm(6), nn.Tanh())
        self.normalize_mfcc = nn.Sequential(nn.LayerNorm(20), nn.Tanh())

        self.conv1 = nn.Conv1d(in_channels=26, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.ln1 = nn.LayerNorm([32])
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.ln2 = nn.LayerNorm([32])
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.ln3 = nn.LayerNorm([32])

    def forward(self, x):
        x_6 = self.normalize_6(x[:, :, :6]).transpose(1, 2)
        x_mfcc = self.normalize_mfcc(x[:, :, 6:]).transpose(1, 2)
        x = torch.cat([x_mfcc, x_6], dim=1)

        h1 = self.conv1(x)
        h1 = torch.relu(self.ln1(h1.transpose(1, 2))).transpose(1, 2)
        h1 = torch.max_pool1d(h1, kernel_size=9, stride=1, padding=4)
        h2 = self.conv2(h1)
        h2 = torch.relu(self.ln2(h2.transpose(1, 2))).transpose(1, 2)
        h2 = torch.max_pool1d(h2, kernel_size=9, stride=1, padding=4)
        h3 = self.conv3(h2)
        h3 = torch.relu(self.ln3(h3.transpose(1, 2))).transpose(1, 2)
        h3 = torch.max_pool1d(h3, kernel_size=9, stride=1, padding=4)

        out_all = torch.cat([h1, h2, h3, x], dim=1)

        return h3.transpose(1, 2), out_all.transpose(1, 2)


class MusicEncoderNoPool(nn.Module):
    def __init__(self):
        super(MusicEncoderNoPool, self).__init__()

        self.normalize_6 = nn.Sequential(nn.LayerNorm(6), nn.Tanh())
        self.normalize_mfcc = nn.Sequential(nn.LayerNorm(20), nn.Tanh())

        self.conv1_x6 = nn.Conv1d(in_channels=6,out_channels=60,kernel_size=25,padding=12,groups=6)
        self.conv1_xmfcc = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=25, padding=12)
        self.ln1 = nn.LayerNorm([80])
        self.conv2 = nn.Conv1d(in_channels=80, out_channels=32, kernel_size=25, stride=1, padding=12)
        self.ln2 = nn.LayerNorm([32])

    def forward(self, x):
        x_6 = torch.sigmoid(self.normalize_6(x[:, :, :6]).transpose(1, 2))
        x_mfcc = torch.sigmoid(self.normalize_mfcc(x[:, :, 6:]).transpose(1, 2))

        h1_x6 = self.conv1_x6(x_6)
        h1_mfcc = self.conv1_xmfcc(x_mfcc)
        h1 = torch.cat([h1_x6,h1_mfcc],dim=1)
        h1 = torch.relu(self.ln1(h1.transpose(1, 2))).transpose(1, 2)
        h1 = torch.avg_pool1d(h1,kernel_size=3,stride=1,padding=1)

        h2 = self.conv2(h1)
        h2 = torch.sigmoid(self.ln2(h2.transpose(1, 2))).transpose(1, 2)

        return torch.cat([h1,h2,x_6,x_mfcc],dim=1).transpose(1, 2), h2.transpose(1, 2)


class MusicEncoderFeatureSkip(nn.Module):
    def __init__(self):
        super(MusicEncoderFeatureSkip, self).__init__()

        self.normalize_6 = nn.Sequential(nn.LayerNorm(6), nn.Tanh())
        self.normalize_mfcc = nn.Sequential(nn.LayerNorm(20), nn.Tanh())


    def forward(self, x):
        x_6 = torch.sigmoid(self.normalize_6(x[:, :, :6]).transpose(1, 2))
        x_mfcc = torch.sigmoid(self.normalize_mfcc(x[:, :, 6:]).transpose(1, 2))

        return torch.cat([x_6,x_mfcc],dim=1).transpose(1, 2)

if __name__ == '__main__':
    music_encoder = MusicEncoderPool(1000).cuda()
    x = torch.randn([10, 1000, 622]).cuda()
    hx = music_encoder(x)
    print(hx.size())
