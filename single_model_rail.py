# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/11 10:59
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class Feature_Extraction(nn.Module):
    """
    Encoder network for rail image input list and gaf image input list
    """

    def __init__(self, c_dim, normalize=True):
        """
        :param c_dim: output dimension of latent embedding (int)
        :param normalize: whether the input image should be normalized (bool)
        """
        super().__init__()
        self.normalized = normalize
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalized:
                x = normalize_img(x)
            c += self.features(x)


def normalize_img(x):
    """

    :param x: input image (tensor)
    :return: narmalized image
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class Encoder(nn.Module):
    """
    Fusion with transformer for 10 gaf and rail feature
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rail_encoder = Feature_Extraction(512, normalize=True)

    def forward(self, rail_list):
        if self.rail_encoder.normalized:
            rail_list = [normalize_img(image_input) for image_input in rail_list]

            bz, _, h, w = rail_list[0].shape

            rail_channel = rail_list[0].shape[1]

            rail_tensor = torch.stack(rail_list, dim=1).view(bz * self.config.seq_len, rail_channel, h, w)

            rail_features = self.rail_encoder.features.conv1(rail_tensor)
            rail_features = self.rail_encoder.features.bn1(rail_features)
            rail_features = self.rail_encoder.features.relu(rail_features)
            rail_features = self.rail_encoder.features.maxpool(rail_features)

            rail_features = self.rail_encoder.features.layer1(rail_features)
            rail_features = self.rail_encoder.features.layer2(rail_features)
            rail_features = self.rail_encoder.features.layer3(rail_features)
            rail_features = self.rail_encoder.features.layer4(rail_features)

            rail_features = self.rail_encoder.features.avgpool(rail_features)
            rail_features = torch.flatten(rail_features, 1)
            rail_features = rail_features.view(bz, self.config.seq_len, -1)

            rail_features = torch.sum(rail_features, dim=1)

            return rail_features


class Decoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        self.encoder = Encoder(config).to(self.device)

        self.MLP = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        ).to(self.device)

        self.output = nn.Linear(64, 5).to(self.device)

    def forward(self, rail_list):
        rail_features = self.encoder(rail_list)
        rail_mlp = self.MLP(rail_features)
        rail_out_put = self.output(rail_mlp)

        return rail_out_put
