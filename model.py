"""
Two models, both deliberately simple:

1. VentricleNet  - a small U-Net that segments only the ventricle (binary).
   Ventricle segmentation is already close to solved on MS3SEG (baseline
   DSC ~0.90, see Table 8 of the dataset paper), so a lightweight, fast,
   reliable Stage-1 model is enough. Its predicted probability map is what
   Stage-2 conditions on (see data.ventricle_distance_from_prob / train.py) -
   nothing about it needs to be learned jointly with the harder tri-mask task.

2. AnatoUNetPP   - a capacity-matched U-Net++ (nested skip pathways) for the
   4-class tri-mask task. The only architectural difference from a plain
   capacity-matched U-Net++ baseline is one extra input channel carrying the
   Stage-1 ventricle-proximity map. That is the entire novel contribution:
   everything else (encoder/decoder width, normalization, loss) is kept
   identical to the baseline on purpose, so any improvement can be
   attributed to the anatomical prior and not to "more machinery."

Both models are optionally supervised at two decoder depths (deep
supervision) using the *same* Dice+CE loss at each depth - this is a
well-established, low-risk regularizer (nnU-Net uses it by default) and
adds no new loss family, unlike the previous VentiMorph-RelNet design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def _up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class VentricleNet(nn.Module):
    """Plain 4-level U-Net, single-channel FLAIR input, binary ventricle output."""

    def __init__(self, base: int = 24):
        super().__init__()
        c = [base, base * 2, base * 4, base * 6, base * 8]
        self.enc0 = ConvBlock(1, c[0])
        self.enc1 = ConvBlock(c[0], c[1])
        self.enc2 = ConvBlock(c[1], c[2])
        self.enc3 = ConvBlock(c[2], c[3])
        self.bottleneck = ConvBlock(c[3], c[4])
        self.pool = nn.MaxPool2d(2)

        self.dec3 = ConvBlock(c[4] + c[3], c[3])
        self.dec2 = ConvBlock(c[3] + c[2], c[2])
        self.dec1 = ConvBlock(c[2] + c[1], c[1])
        self.dec0 = ConvBlock(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([_up(b, e3.shape[-2:]), e3], 1))
        d2 = self.dec2(torch.cat([_up(d3, e2.shape[-2:]), e2], 1))
        d1 = self.dec1(torch.cat([_up(d2, e1.shape[-2:]), e1], 1))
        d0 = self.dec0(torch.cat([_up(d1, e0.shape[-2:]), e0], 1))
        return self.head(d0)


class AnatoUNetPP(nn.Module):
    """Capacity-matched (~6M params at base=32) nested U-Net++ for the 4-class
    tri-mask task. `in_channels`=9 reproduces the plain multimodal 2.5D
    baseline exactly; `in_channels`=10 adds the ventricle-proximity channel.
    """

    def __init__(self, in_channels: int = 10, num_classes: int = 4, base: int = 32,
                 deep_supervision: bool = True):
        super().__init__()
        c = [base, base * 2, base * 4, base * 6, base * 8]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)

        self.x00 = ConvBlock(in_channels, c[0])
        self.x10 = ConvBlock(c[0], c[1])
        self.x20 = ConvBlock(c[1], c[2])
        self.x30 = ConvBlock(c[2], c[3])
        self.x40 = ConvBlock(c[3], c[4])

        self.x01 = ConvBlock(c[0] + c[1], c[0])
        self.x11 = ConvBlock(c[1] + c[2], c[1])
        self.x21 = ConvBlock(c[2] + c[3], c[2])
        self.x31 = ConvBlock(c[3] + c[4], c[3])

        self.x02 = ConvBlock(c[0] * 2 + c[1], c[0])
        self.x12 = ConvBlock(c[1] * 2 + c[2], c[1])
        self.x22 = ConvBlock(c[2] * 2 + c[3], c[2])

        self.x03 = ConvBlock(c[0] * 3 + c[1], c[0])
        self.x13 = ConvBlock(c[1] * 3 + c[2], c[1])

        self.x04 = ConvBlock(c[0] * 4 + c[1], c[0])

        heads = 4 if deep_supervision else 1
        self.heads = nn.ModuleList([nn.Conv2d(c[0], num_classes, 1) for _ in range(heads)])

    def forward(self, x):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x01 = self.x01(torch.cat([x00, _up(x10, x00.shape[-2:])], 1))

        x20 = self.x20(self.pool(x10))
        x11 = self.x11(torch.cat([x10, _up(x20, x10.shape[-2:])], 1))
        x02 = self.x02(torch.cat([x00, x01, _up(x11, x00.shape[-2:])], 1))

        x30 = self.x30(self.pool(x20))
        x21 = self.x21(torch.cat([x20, _up(x30, x20.shape[-2:])], 1))
        x12 = self.x12(torch.cat([x10, x11, _up(x21, x10.shape[-2:])], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, _up(x12, x00.shape[-2:])], 1))

        x40 = self.x40(self.pool(x30))
        x31 = self.x31(torch.cat([x30, _up(x40, x30.shape[-2:])], 1))
        x22 = self.x22(torch.cat([x20, x21, _up(x31, x20.shape[-2:])], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, _up(x22, x10.shape[-2:])], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, _up(x13, x00.shape[-2:])], 1))

        if not self.deep_supervision:
            return {"logits": self.heads[0](x04), "aux_logits": []}

        main = self.heads[0](x04)
        aux = [self.heads[1](x03), self.heads[2](x02), self.heads[3](x01)]
        return {"logits": main, "aux_logits": aux}


if __name__ == "__main__":
    v = VentricleNet()
    a = AnatoUNetPP(in_channels=10)
    dummy_flair = torch.randn(1, 1, 256, 256)
    dummy_full = torch.randn(1, 10, 256, 256)
    print("VentricleNet params:", sum(p.numel() for p in v.parameters()))
    print("VentricleNet out:", v(dummy_flair).shape)
    print("AnatoUNetPP params:", sum(p.numel() for p in a.parameters()))
    out = a(dummy_full)
    print("AnatoUNetPP main logits:", out["logits"].shape, "aux:", [t.shape for t in out["aux_logits"]])
