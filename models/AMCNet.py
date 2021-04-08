import torch
from torch import nn
from models.MusicEncoder import MusicEncoderNoPool, MusicEncoderPool
from models.PoseEncoder import PoseEncoderNoPool, PoseEncoderDr


class AMCNet_shallow(nn.Module):
    def __init__(self):
        super(AMCNet_shallow, self).__init__()

        self.music_encoder = MusicEncoderNoPool()
        self.pose_encoder = PoseEncoderNoPool()

        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=15, stride=1, padding=7)
        self.ln1 = nn.LayerNorm([128])
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=15, stride=1, padding=7)
        self.ln2 = nn.LayerNorm([32])
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=15, stride=1, padding=7)
        self.ln3 = nn.LayerNorm([32])


        self.correspondFC = nn.Sequential(
            nn.Linear(250, 16),
            # nn.ReLU(),
            # nn.Linear(16, 1),
            # nn.Sigmoid()
        )

    def forward(self, x, y):

        hx, hx2 = self.music_encoder(x)
        hy, hy2 = self.pose_encoder(y)
        feature_cat = torch.cat([hx, hy], dim=2)
        L3score = self.correspondFC(feature_cat)

        hx2 = torch.unsqueeze(hx2,dim=2)
        hy2 = torch.unsqueeze(hy2,dim=3)
        feature_mul = hx2.mul(hy2)
        feature_mul = torch.flatten(feature_mul,start_dim=2)

        h1 = self.conv1(feature_mul.transpose(1, 2))
        h1 = torch.relu(self.ln1(h1.transpose(1, 2)).transpose(1, 2))
        h1 = torch.avg_pool1d(h1, kernel_size=3, stride=5, padding=1)

        h2 = self.conv2(h1)
        h2 = torch.relu(self.ln2(h2.transpose(1, 2)).transpose(1, 2))
        h2 = torch.avg_pool1d(h2, kernel_size=3, stride=5, padding=1)

        h3 = self.conv3(h2)
        #h3 = torch.relu(self.ln3(h3.transpose(1, 2)).transpose(1, 2))
        Dc_out = torch.avg_pool1d(h3, kernel_size=3, stride=5, padding=1)
        Dc_out = torch.sigmoid(Dc_out)

        return L3score, Dc_out.transpose(1, 2), feature_cat, h1.transpose(1, 2)
