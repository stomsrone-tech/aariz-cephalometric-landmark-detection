"""
U-Net for heatmap-regression cephalometric landmark detection.
Input: (B, 3, image_size, image_size) RGB image
Output: (B, 29, heatmap_size, heatmap_size) per-landmark heatmaps

Deliberately a standard, well-understood U-Net (not a novel architecture) — this is the
baseline model per the project plan; HRNet comes later as the upgrade (Phase 5).
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_landmarks=29, base_channels=32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = conv_block(3, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.enc4 = conv_block(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(c * 8, c * 16)

        # Decoder — image_size=512 -> heatmap_size=128 means we only need to upsample
        # back to 1/4 resolution, not all the way to full input resolution.
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = conv_block(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        # Encoder: 512->256->128->64->32 (4 pools, bottleneck at 32).
        # Decoder: 32->64->128->256, i.e. output at 256x256 = heatmap_size, with c*2 channels.
        # This is the heatmap_size=256 upgrade (was 128) to reduce the resolution-dependent
        # quantization error identified in the device breakdown (Phase 7) — halving the
        # decode-uncertainty-per-mm for high-resolution source images like Rotograph EVO
        # and ISBI 2015.

        self.out_conv = nn.Conv2d(c * 2, n_landmarks, 1)

    def forward(self, x):
        e1 = self.enc1(x)            # 512
        e2 = self.enc2(self.pool(e1))  # 256
        e3 = self.enc3(self.pool(e2))  # 128
        e4 = self.enc4(self.pool(e3))  # 64
        b = self.bottleneck(self.pool(e4))  # 32

        d4 = self.up4(b)             # 64
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)            # 128
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)            # 256
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        out = self.out_conv(d2)      # (B, n_landmarks, 256, 256)
        return out


class UNetLegacy128(nn.Module):
    """Reconstruction of the ORIGINAL U-Net architecture (before the 128->256 heatmap
    resolution upgrade in Phase 7) -- exists only so the checkpoint saved from that
    earlier training run (model_output/best_model.pth) can still be loaded for
    evaluation. Do not use this for new training; UNet (above) is the current architecture.
    Output: (B, n_landmarks, 128, 128).
    """
    def __init__(self, n_landmarks=29, base_channels=32):
        super().__init__()
        c = base_channels

        self.enc1 = conv_block(3, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.enc4 = conv_block(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = conv_block(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)

        self.out_conv = nn.Conv2d(c * 4, n_landmarks, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        out = self.out_conv(d3)
        return out


if __name__ == "__main__":
    # Smoke test: confirm output shape matches the new heatmap_size=256 target.
    model = UNet(n_landmarks=29)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    expected = (2, 29, 256, 256)
    assert tuple(y.shape) == expected, f"Shape mismatch! Got {tuple(y.shape)}, expected {expected}"
    print("Shape check passed — matches AarizDataset(heatmap_size=256) exactly.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
