import torch
from torch import nn

from models.MusicEncoder import MusicEncoderNoPool, MusicEncoderFeatureSkip
from models.PoseDecoder import PoseDecoderLSTM, PoseDecoderTCN
from models.PoseEncoder import PoseEncoderStyle, PoseEncoderDr


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.music_encoder = MusicEncoderNoPool()
        # self.style_encoder = PoseEncoderStyle()
        self.pose_decoder = PoseDecoderTCN(138)
        self.out_lstm = PoseDecoderLSTM(20)

    def forward(self, x, y):
        hx_all, hx = self.music_encoder(x)
        hx_all = torch.rand_like(hx_all)
        y = self.pose_decoder(hx_all)
        y = self.out_lstm(y)
        return y, hx_all


class Generator_Sampling(nn.Module):
    def __init__(self):
        super(Generator_Sampling, self).__init__()

        self.music_encoder = MusicEncoderNoPool()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=138, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.Tanh()
        )

        self.out_lstm = PoseDecoderLSTM(128)

    def forward(self, x, y):
        hx, _ = self.music_encoder(x)
        h = self.encoder(hx.transpose(1,2))
        y = self.decoder(h).transpose(1,2)
        y = self.out_lstm(y)
        return y, h.transpose(1,2)


if __name__ == '__main__':
    G = Generator_Sampling().cuda()
    x = torch.randn([10, 1001, 26]).cuda()
    y= G(x)

    print(y.size())
