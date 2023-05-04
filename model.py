# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/20 12:10
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class Feature_Extraction(nn.Module):
    """
    提取图像特征
    """

    def __init__(self, c_dim, normalize=True):
        """
        :param c_dim:
        :param normalize: 执行正则化
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


class Self_Attention(nn.Module):
    """
    多头自注意力机制
    """

    def __init__(self, n_embed, n_head, attn_drop, resid_drop):
        super().__init__()
        assert n_embed % n_head == 0
        self.key = nn.Linear(n_embed, n_embed)
        self.query = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)

        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        self.proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,T,nh,hs)->(B,nh,T,hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B,nh,T,hs)*(B,nh,hs,T)->(B,nh,T,T)
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)

        # (B,nh,T,T)*(B,nh,T,hs)->(B,nh,T,hs)
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))

        return y


class Block(nn.Module):
    """
    Transformer模块
    """

    def __init__(self, n_embed, n_head, block_exp, attn_drop, resid_drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = Self_Attention(n_embed, n_head, attn_drop, resid_drop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, block_exp * n_embed),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embed, n_embed),
            nn.Dropout(resid_drop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class Pre_embedding(nn.Module):
    def __init__(self, n_embed, n_head, block_exp, n_layer, seq_len, embed_drop, attn_drop,
                 resid_drop, config):
        super().__init__()
        self.n_embed = n_embed
        self.seq_len = seq_len
        self.config = config

        self.drop = nn.Dropout(embed_drop)

        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_exp, attn_drop, resid_drop)
                                      for layer in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed)

        self.blocks_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.blocks_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                full_param_name = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(full_param_name)

                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(full_param_name)

                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    decay.add(full_param_name)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_group = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {'params': [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
        ]

        return optim_group

    def forward(self, gaf_tensor, rail_tensor):
        """
        :param gaf_tensor: B*10*seq_len,C,H,W
        :param rail_tensor: B*seq_len,C,H,W
        :return: gaf_out_tensor, rail_out_tensor
        """

        bz = rail_tensor.shape[0] // self.seq_len
        h, w = rail_tensor.shape[2:4]

        gaf_tensor = gaf_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        rail_tensor = rail_tensor.view(bz, self.seq_len, -1, h, w)

        token_embedding = torch.cat([gaf_tensor, rail_tensor], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        token_embedding = token_embedding.view(bz, -1, self.n_embed)

        x = self.drop(token_embedding)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.view(bz, (self.config.n_views + 1) * self.seq_len, -1, self.n_embed)
        x = x.permute(0, 1, 3, 2).contiguous()

        gaf_out_tensor = x[:, :self.config.n_views * self.seq_len, :, :].contiguous().view(
            bz * self.config.n_views * self.seq_len, -1, h, w)
        rail_out_tensor = x[:, self.config.n_views * self.seq_len:, :, :].contiguous().view(
            bz * self.seq_len, -1, h, w)

        return gaf_out_tensor, rail_out_tensor


class Encoder(nn.Module):
    """
    融合两种模态
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.gaf_encoder = Feature_Extraction(512, normalize=True)
        self.rail_encoder = Feature_Extraction(512, normalize=True)

        self.trans1 = Pre_embedding(n_embed=64,
                                    n_head=config.n_head,
                                    block_exp=config.block_exp,
                                    n_layer=config.n_layer,
                                    seq_len=config.seq_len,
                                    embed_drop=config.embed_drop,
                                    attn_drop=config.attn_drop,
                                    resid_drop=config.resid_drop,
                                    config=config)
        self.trans2 = Pre_embedding(n_embed=128,
                                    n_head=config.n_head,
                                    block_exp=config.block_exp,
                                    n_layer=config.n_layer,
                                    seq_len=config.seq_len,
                                    embed_drop=config.embed_drop,
                                    attn_drop=config.attn_drop,
                                    resid_drop=config.resid_drop,
                                    config=config)
        self.trans3 = Pre_embedding(n_embed=256,
                                    n_head=config.n_head,
                                    block_exp=config.block_exp,
                                    n_layer=config.n_layer,
                                    seq_len=config.seq_len,
                                    embed_drop=config.embed_drop,
                                    attn_drop=config.attn_drop,
                                    resid_drop=config.resid_drop,
                                    config=config)
        self.trans4 = Pre_embedding(n_embed=512,
                                    n_head=config.n_head,
                                    block_exp=config.block_exp,
                                    n_layer=config.n_layer,
                                    seq_len=config.seq_len,
                                    embed_drop=config.embed_drop,
                                    attn_drop=config.attn_drop,
                                    resid_drop=config.resid_drop,
                                    config=config)

    def forward(self, gaf_list, rail_list):
        if self.gaf_encoder.normalized:
            gaf_list = [normalize_img(image_input_g) for image_input_g in gaf_list]
            rail_list = [normalize_img(image_input_r) for image_input_r in rail_list]

        bz, _, h, w = rail_list[0].shape

        gaf_channel = gaf_list[0].shape[1]
        rail_channel = rail_list[0].shape[1]

        self.config.n_views = len(gaf_list) // self.config.seq_len

        gaf_tensor = torch.stack(gaf_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, gaf_channel, h,
                                                       w)
        rail_tensor = torch.stack(rail_list, dim=1).view(bz * self.config.seq_len, rail_channel, h, w)

        gaf_features = self.gaf_encoder.features.conv1(gaf_tensor)
        gaf_features = self.gaf_encoder.features.bn1(gaf_features)
        gaf_features = self.gaf_encoder.features.relu(gaf_features)
        gaf_features = self.gaf_encoder.features.maxpool(gaf_features)

        rail_features = self.rail_encoder.features.conv1(rail_tensor)
        rail_features = self.rail_encoder.features.bn1(rail_features)
        rail_features = self.rail_encoder.features.relu(rail_features)
        rail_features = self.rail_encoder.features.maxpool(rail_features)

        """
        fusion at (B,64,64,64)
        """
        gaf_features = self.gaf_encoder.features.layer1(gaf_features)
        rail_features = self.rail_encoder.features.layer1(rail_features)

        gaf_embed_layer1 = self.avgpool(gaf_features)
        rail_embed_layer1 = self.avgpool(rail_features)
        gaf_features_layer1, rail_features_layer1 = self.trans1(gaf_embed_layer1, rail_embed_layer1)
        gaf_features_layer1 = F.interpolate(gaf_features_layer1, scale_factor=16, mode="bilinear")  # 64->512
        rail_features_layer1 = F.interpolate(rail_features_layer1, scale_factor=16, mode="bilinear")
        gaf_features = gaf_features + gaf_features_layer1
        rail_features = rail_features + rail_features_layer1

        """
        fusion at (B,128,32,32)
        """
        gaf_features = self.gaf_encoder.features.layer2(gaf_features)
        rail_features = self.rail_encoder.features.layer2(rail_features)

        gaf_embed_layer2 = self.avgpool(gaf_features)
        rail_embed_layer2 = self.avgpool(rail_features)
        gaf_features_layer2, rail_features_layer2 = self.trans2(gaf_embed_layer2, rail_embed_layer2)
        gaf_features_layer2 = F.interpolate(gaf_features_layer2, scale_factor=8, mode="bilinear")  # 128->512
        rail_features_layer2 = F.interpolate(rail_features_layer2, scale_factor=8, mode="bilinear")
        gaf_features = gaf_features + gaf_features_layer2
        rail_features = rail_features + rail_features_layer2

        """
        fusion at (B,256,16,16)
        """
        gaf_features = self.gaf_encoder.features.layer3(gaf_features)
        rail_features = self.rail_encoder.features.layer3(rail_features)

        gaf_embed_layer3 = self.avgpool(gaf_features)
        rail_embed_layer3 = self.avgpool(rail_features)
        gaf_features_layer3, rail_features_layer3 = self.trans3(gaf_embed_layer3, rail_embed_layer3)
        gaf_features_layer3 = F.interpolate(gaf_features_layer3, scale_factor=4, mode="bilinear")  # 256->512
        rail_features_layer3 = F.interpolate(rail_features_layer3, scale_factor=4, mode="bilinear")
        gaf_features = gaf_features + gaf_features_layer3
        rail_features = rail_features + rail_features_layer3

        """
        fusion at (B,512,8,8)
        """
        gaf_features = self.gaf_encoder.features.layer4(gaf_features)
        rail_features = self.rail_encoder.features.layer4(rail_features)

        gaf_embed_layer4 = self.avgpool(gaf_features)
        rail_embed_layer4 = self.avgpool(rail_features)
        gaf_features_layer4, rail_features_layer4 = self.trans4(gaf_embed_layer4, rail_embed_layer4)
        gaf_features_layer4 = F.interpolate(gaf_features_layer4, scale_factor=2, mode="bilinear")  # 256->512
        rail_features_layer4 = F.interpolate(rail_features_layer4, scale_factor=2, mode="bilinear")
        gaf_features = gaf_features + gaf_features_layer4
        rail_features = rail_features + rail_features_layer4

        gaf_features = self.gaf_encoder.features.avgpool(gaf_features)
        gaf_features = torch.flatten(gaf_features, 1)
        gaf_features = gaf_features.view(bz, self.config.n_views * self.config.seq_len, -1)
        rail_features = self.rail_encoder.features.avgpool(rail_features)
        rail_features = torch.flatten(rail_features, 1)
        rail_features = rail_features.view(bz, self.config.seq_len, -1)

        fusion_features = torch.cat([gaf_features, rail_features], dim=1)
        fusion_features = torch.sum(fusion_features, dim=1)

        return fusion_features


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

    def forward(self, gaf_list, rail_list):
        fused_features = self.encoder(gaf_list, rail_list)
        mlp = self.MLP(fused_features)
        out_put = self.output(mlp)

        return out_put
