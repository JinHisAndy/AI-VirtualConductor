import torch
from torch import nn

from models.PoseEncoder import PoseEncoderDr
from models.MusicEncoder import MusicEncoderNoPool
from models.AMCNet import AMCNet_shallow


class RealFakeDiscriminator(nn.Module):
    def __init__(self):
        super(RealFakeDiscriminator, self).__init__()

        self.music_encoder = MusicEncoderNoPool()
        self.conv1_group = nn.Conv1d(in_channels=20, out_channels=80, kernel_size=15, stride=1, padding=7, groups=20)
        self.conv1_nogroup = nn.Conv1d(in_channels=20, out_channels=80, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(in_channels=160, out_channels=128, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=15, stride=1, padding=7)

        self.pose_encoder = PoseEncoderDr()

        self.out = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, y):
        y = y.transpose(1, 2)
        h1_group = torch.relu(self.conv1_group(y))
        h1_nogroup = torch.relu(self.conv1_nogroup(y))
        h1 = torch.cat([h1_group, h1_nogroup], dim=1)
        h1 = torch.max_pool1d(h1, kernel_size=5, stride=5)

        h2 = self.conv2(h1)
        h2 = torch.relu(h2)
        h2 = torch.max_pool1d(h2, kernel_size=5, stride=5)

        h3 = self.conv3(h2)
        h3 = torch.relu(h3)

        h1 = h1.transpose(1, 2)
        h2 = h2.transpose(1, 2)
        h3 = h3.transpose(1, 2)

        real_fake = self.out(h3)

        return real_fake, h1


class DoubleAMCDiscriminator(nn.Module):
    def __init__(self):
        super(DoubleAMCDiscriminator, self).__init__()

        self.ACM_Freeze = AMCNet_shallow()
        self.ACM_unFreeze = AMCNet_shallow()

        self.conv1 = nn.Conv1d(in_channels=500, out_channels=160, kernel_size=15, stride=1, padding=7, groups=2)
        self.conv2 = nn.Conv1d(in_channels=160, out_channels=64, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=15, stride=1, padding=7)

        self.out = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, y):
        _, _, L3_feature, _ = self.ACM_Freeze(x, y)
        _, _, Dr_feature, _ = self.ACM_unFreeze(x, y)
        cat_feature = torch.cat([L3_feature, Dr_feature], dim=2).transpose(1, 2)

        h1 = self.conv1(cat_feature)
        h1 = torch.relu(h1)
        h1 = torch.max_pool1d(h1, kernel_size=5, stride=5)

        h2 = self.conv2(h1)
        h2 = torch.relu(h2)
        h2 = torch.max_pool1d(h2, kernel_size=5, stride=5)

        h3 = self.conv3(h2)
        h3 = torch.relu(h3)
        h3 = torch.max_pool1d(h3, kernel_size=5, stride=5)

        real_fake = self.out(h3.transpose(1, 2))

        return real_fake, cat_feature.transpose(1, 2)
