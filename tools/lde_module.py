import torch
import torch.nn as nn


class DMlpPlug(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        self.proj_in = nn.Conv2d(3, 36, kernel_size=3, padding=1)
        self.upsample_path = nn.Sequential(
            nn.Conv2d(36, 36 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),  
            nn.Conv2d(36, 36, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.dmlp = DMlp(dim=36, growth_rate=2.0)
        self.downsample = nn.Upsample(scale_factor=1 / scale_factor,
                                      mode='bilinear') if scale_factor > 1 else nn.Identity()

        self.proj_out = nn.Conv2d(36, 3, kernel_size=3, padding=1)

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x
        x = self.proj_in(x)
        if self.scale_factor > 1:
            upsampled = self.upsample_path(x)
            upsampled = self.downsample(upsampled) 
            x = x + self.alpha * upsampled 

        x = self.dmlp(x)
        x = self.proj_out(x)

        return x + identity 

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x