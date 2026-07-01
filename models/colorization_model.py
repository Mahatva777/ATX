"""
Colorization model: single-channel TIR 512x512 @100m -> 3-channel RGB 512x512 @100m.

Note: driver.py actually produces 512x512 patches (not 256x256 as the README
describes). The UNetGenerator below uses a 9-level U-Net so the bottleneck
remains 1x1 at 512x512 input (an 8-level U-Net would bottleneck at 2x2,
leaving unused capacity and potentially worse compression).

Pix2Pix-style conditional GAN:
  - Generator: U-Net (encoder-decoder with skip connections). Skip
    connections matter here specifically because the task must "Preserve
    Semantic Integrity" (per the PS) -- we need fine structural edges from
    the input TIR (roads, building outlines) to pass through untouched
    rather than be re-hallucinated from a bottleneck.
  - Discriminator: PatchGAN (70x70 receptive field) -- judges local
    patches as real/fake rather than the whole image, which keeps the
    adversarial signal focused on local color/texture realism instead of
    global structure (global structure is already the generator's job via
    skip connections + L1).
  - Loss: L1 (structural fidelity, weighted heavily) + adversarial
    (realism) + optional semantic-constraint loss from
    semantic_constraint.py.
"""
import torch
import torch.nn as nn


def conv_block(in_c, out_c, down=True, use_norm=True, use_dropout=False, act="leaky"):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_norm))
    else:
        layers.append(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=not use_norm))
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_c))
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.LeakyReLU(0.2, inplace=True) if act == "leaky" else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """
    Input:  (B, 1, 512, 512)  -- single-channel TIR (actual driver.py patch size)
    Output: (B, 3, 512, 512)  -- RGB, values in [-1, 1] (tanh)

    9-level U-Net so the bottleneck is 1x1 at 512x512 input.
    (8 levels would give a 2x2 bottleneck — suboptimal for compression.)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base: int = 64):
        super().__init__()
        # Encoder: 512->256->128->64->32->16->8->4->2->1
        self.e1 = nn.Sequential(nn.Conv2d(in_channels, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))  # 256
        self.e2 = conv_block(base, base * 2)          # 128
        self.e3 = conv_block(base * 2, base * 4)      # 64
        self.e4 = conv_block(base * 4, base * 8)      # 32
        self.e5 = conv_block(base * 8, base * 8)      # 16
        self.e6 = conv_block(base * 8, base * 8)      # 8
        self.e7 = conv_block(base * 8, base * 8)      # 4
        self.e8 = conv_block(base * 8, base * 8)      # 2  ← extra level for 512 input
        self.bottleneck = nn.Sequential(nn.Conv2d(base * 8, base * 8, 4, 2, 1), nn.ReLU(inplace=True))  # 1

        # Decoder (skip connections double the channel count on the way up)
        self.d8 = conv_block(base * 8,  base * 8,  down=False, use_dropout=True, act="relu")
        self.d7 = conv_block(base * 16, base * 8,  down=False, use_dropout=True, act="relu")
        self.d6 = conv_block(base * 16, base * 8,  down=False, use_dropout=True, act="relu")
        self.d5 = conv_block(base * 16, base * 8,  down=False, act="relu")
        self.d4 = conv_block(base * 16, base * 8,  down=False, act="relu")
        self.d3 = conv_block(base * 16, base * 4,  down=False, act="relu")
        self.d2 = conv_block(base * 8,  base * 2,  down=False, act="relu")
        self.d1 = conv_block(base * 4,  base,      down=False, act="relu")
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        b  = self.bottleneck(e8)

        d8 = self.d8(b)
        d7 = self.d7(torch.cat([d8, e8], dim=1))
        d6 = self.d6(torch.cat([d7, e7], dim=1))
        d5 = self.d5(torch.cat([d6, e6], dim=1))
        d4 = self.d4(torch.cat([d5, e5], dim=1))
        d3 = self.d3(torch.cat([d4, e4], dim=1))
        d2 = self.d2(torch.cat([d3, e3], dim=1))
        d1 = self.d1(torch.cat([d2, e2], dim=1))
        out = self.final(torch.cat([d1, e1], dim=1))
        return out

class PatchGANDiscriminator(nn.Module):
    """
    Takes the TIR input concatenated with either the real or fake RGB
    (conditional GAN) and outputs a grid of real/fake patch scores.
    Works for both 256x256 and 512x512 inputs (output grid scales with input).
    Input: (B, 1 + 3, H, W) -> Output: (B, 1, ~30, ~30) for H=256, ~62x62 for H=512.
    """

    def __init__(self, in_channels: int = 1 + 3, base: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            conv_block(base, base * 2),
            conv_block(base * 2, base * 4),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1), nn.InstanceNorm2d(base * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, tir, rgb):
        x = torch.cat([tir, rgb], dim=1)
        return self.model(x)


if __name__ == "__main__":
    g = UNetGenerator()
    d = PatchGANDiscriminator()
    tir = torch.randn(2, 1, 512, 512)
    rgb_fake = g(tir)
    assert rgb_fake.shape == (2, 3, 512, 512), rgb_fake.shape
    score = d(tir, rgb_fake)
    print("Generator OK:", rgb_fake.shape)
    print("Discriminator OK:", score.shape)
