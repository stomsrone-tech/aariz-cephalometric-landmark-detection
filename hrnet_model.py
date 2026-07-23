"""
HRNet-style heatmap regression model — the Phase 5 comparison architecture.

Unlike U-Net (single encoder-decoder path with skip connections), this maintains
parallel branches at multiple resolutions throughout the network, with repeated
cross-resolution fusion — the core architectural idea behind HRNet [Sun et al. 2019].

Note: this is a "HRNet-style" network, not a literal reimplementation of the full
published HRNet-W32/W48 spec (which uses many more repeated fusion modules per stage
and is substantially larger/slower to train). This is a deliberate simplification for
tractability within a single-session Colab training run, while preserving the key
architectural property (parallel multi-resolution representation + fusion) that
distinguishes it from the U-Net baseline. State this precisely in the Methods section
rather than claiming exact correspondence to the original HRNet paper.

Input: (B, 3, 512, 512)
Output: (B, 29, 256, 256) — same output convention as unet_model.py, so the rest of the
pipeline (aariz_dataset.py, train_eval.py) works unchanged with either architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


def make_branch(channels, n_blocks=2):
    return nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])


class HRNetStyle(nn.Module):
    def __init__(self, n_landmarks=29, base_channels=32):
        super().__init__()
        c = base_channels  # 32

        # Stem: single stride-2 conv, 512 -> 256. This makes the highest-resolution
        # branch land exactly on heatmap_size=256, avoiding any final upsample step.
        self.stem = conv_bn_relu(3, c * 2, stride=2)  # 512 -> 256, 64 channels

        # Stage 1: single branch at 256 res
        self.stage1 = make_branch(c * 2, n_blocks=2)  # 64 channels @ 256

        # Stage 2: split into branch1 (256res, 64ch) and branch2 (128res, 128ch)
        self.branch2_down = conv_bn_relu(c * 2, c * 4, stride=2)  # 256 -> 128, 128 channels
        self.stage2_b1 = make_branch(c * 2, n_blocks=2)
        self.stage2_b2 = make_branch(c * 4, n_blocks=2)
        # Fusion convs (applied after resizing, to adjust channel counts where needed)
        self.fuse2to1 = nn.Conv2d(c * 4, c * 2, 1)  # branch2 -> branch1 channel count, after upsample
        self.fuse1to2 = conv_bn_relu(c * 2, c * 4, stride=2)  # branch1 -> branch2 resolution+channels

        # Stage 3: add branch3 (64res, 256ch), derived from branch2
        self.branch3_down = conv_bn_relu(c * 4, c * 8, stride=2)  # 128 -> 64, 256 channels
        self.stage3_b1 = make_branch(c * 2, n_blocks=2)
        self.stage3_b2 = make_branch(c * 4, n_blocks=2)
        self.stage3_b3 = make_branch(c * 8, n_blocks=2)
        self.fuse2to1_v2 = nn.Conv2d(c * 4, c * 2, 1)
        self.fuse3to1_v2 = nn.Conv2d(c * 8, c * 2, 1)

        # Head: fuse all 3 branches (upsampled to 256 res) and produce the heatmaps
        self.head = nn.Sequential(
            nn.Conv2d(c * 2 * 3, c * 2, 3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, n_landmarks, 1),
        )

    def forward(self, x):
        x = self.stem(x)          # (B, 64, 256, 256)
        b1 = self.stage1(x)       # (B, 64, 256, 256)

        # --- Stage 2: two parallel branches with fusion ---
        b2 = self.branch2_down(b1)  # (B, 128, 128, 128)
        b1 = self.stage2_b1(b1)
        b2 = self.stage2_b2(b2)
        b1_fused = b1 + self.fuse2to1(F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False))
        b2_fused = b2 + self.fuse1to2(b1)
        b1, b2 = b1_fused, b2_fused

        # --- Stage 3: three parallel branches with fusion ---
        b3 = self.branch3_down(b2)  # (B, 256, 64, 64)
        b1 = self.stage3_b1(b1)
        b2 = self.stage3_b2(b2)
        b3 = self.stage3_b3(b3)
        b1 = b1 + self.fuse2to1_v2(F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False)) \
                + self.fuse3to1_v2(F.interpolate(b3, size=b1.shape[-2:], mode="bilinear", align_corners=False))

        # --- Head: fuse all resolutions into the final heatmap output ---
        b2_up = F.interpolate(b2, size=b1.shape[-2:], mode="bilinear", align_corners=False)
        b3_up = F.interpolate(b3, size=b1.shape[-2:], mode="bilinear", align_corners=False)
        # Project b2_up (128ch) and b3_up (256ch) down to 64ch each so concatenation with
        # b1 (64ch) gives a consistent 192ch input to the head.
        fused = torch.cat([
            b1,
            self.fuse2to1(b2_up),
            self.fuse3to1_v2(b3_up),
        ], dim=1)

        out = self.head(fused)
        return out


if __name__ == "__main__":
    model = HRNetStyle(n_landmarks=29)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    expected = (2, 29, 256, 256)
    assert tuple(y.shape) == expected, f"Shape mismatch! Got {tuple(y.shape)}, expected {expected}"
    print("Shape check passed.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
