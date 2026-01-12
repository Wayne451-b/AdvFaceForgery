import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from tools.dsa_module import DynamicSpatialAttention


class Adp_spa_freq_attention(nn.Module):
    def __init__(self, feat_dim, patch_size):
        super(Adp_spa_freq_attention, self).__init__()
        self.patch_size = patch_size
        self.feat_dim = feat_dim

        self.freq_weight = nn.Parameter(
            torch.ones((feat_dim, 1, 1, patch_size, patch_size // 2 + 1))
        )
        self.Featue = DynamicSpatialAttention(feat_dim)


    def forward(self, x):
        x = self.Featue(x)
        res = x

        x_patched = rearrange(
            x,
            'b c (h p1) (w p2) -> b c h w p1 p2',
            p1=self.patch_size,
            p2=self.patch_size
        )

        x_freq = torch.fft.rfft2(x_patched.float())
        x_freq_modulated = x_freq * self.freq_weight
        x_patched = torch.fft.irfft2(
            x_freq_modulated,
            s=(self.patch_size, self.patch_size)
        )
        x = rearrange(
            x_patched,
            'b c h w p1 p2 -> b c (h p1) (w p2)',
            p1=self.patch_size,
            p2=self.patch_size
        )

        x = x + res
        return x