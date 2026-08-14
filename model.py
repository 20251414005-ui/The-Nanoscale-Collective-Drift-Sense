"""
model.py — Drift-Sense learned localization model

A small fully-convolutional Siamese network:
  - A shared CNN encoder converts both the (downsampled) reference template
    and the search image into feature maps.
  - The template's FULL spatial feature map is used as a cross-correlation
    kernel against the search image's feature map (true spatial matching,
    not a pooled texture average — see SiameseLocalizer.forward docstring
    for why that distinction mattered here).
  - The raw correlation is normalized by feature magnitude (cosine
    similarity), the same principle as the classical NCC baseline, to
    avoid edge/padding artifacts biasing predictions toward image borders.
  - The predicted location is the peak of that similarity heatmap.

This is the same family of approach as SiamFC (Bertinetto et al., 2016,
"Fully-Convolutional Siamese Networks for Object Tracking") — chosen
because a learned embedding can be trained to be robust to sensor noise
and small rotation, which is exactly where the classical NCC baseline
(see localize.py) was shown empirically to hit a ceiling on highly
periodic DRAM/FinFET patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Shared CNN feature extractor. Keeps stride modest so the output
    feature map still has enough spatial resolution to localize precisely."""

    def __init__(self, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # /2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # /8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SiameseLocalizer(nn.Module):
    def __init__(self, embed_channels=64):
        super().__init__()
        self.encoder = Encoder(out_channels=embed_channels)

    def forward(self, template, search):
        """
        template: (B, 1, Ht, Wt) — the downsampled reference patch
        search:   (B, 1, Hs, Ws) — the full search image
        returns:  (B, 1, ho, wo) similarity heatmap, values roughly in
                  [-1, 1] (cosine-similarity style)

        Uses the template's full spatial feature map as the correlation
        kernel (true cross-correlation / SiamFC-style), not a globally
        pooled vector — a pooled version was confirmed to collapse to
        texture-averaging and fail at chance level.

        The raw correlation is additionally NORMALIZED by the template's
        and each local search window's feature magnitude (cosine
        similarity), matching the same principle as normalized
        cross-correlation (NCC) used in the classical baseline. Without
        this, an unbounded dot product was empirically found to be
        artificially inflated near the search image's borders (a
        zero-padding artifact from the conv layers), pulling predictions
        toward corners regardless of actual pattern content.
        """
        template_feat = self.encoder(template)          # (B, C, ht, wt)
        search_feat = self.encoder(search)               # (B, C, hs, ws)

        B, C, ht, wt = template_feat.shape
        ones_kernel = torch.ones(1, C, ht, wt, device=search_feat.device)

        heatmaps = []
        for b in range(B):
            kernel = template_feat[b:b + 1]                       # (1, C, ht, wt)
            raw = F.conv2d(search_feat[b:b + 1], kernel)           # (1, 1, ho, wo)

            template_norm = torch.sqrt((kernel ** 2).sum() + 1e-8)
            local_energy = F.conv2d(search_feat[b:b + 1] ** 2, ones_kernel)  # (1,1,ho,wo)
            local_norm = torch.sqrt(local_energy + 1e-8)

            normalized = raw / (template_norm * local_norm)
            heatmaps.append(normalized)
        heatmap = torch.cat(heatmaps, dim=0)
        return heatmap


if __name__ == "__main__":
    # quick shape sanity check
    model = SiameseLocalizer()
    template = torch.randn(2, 1, 100, 100)
    search = torch.randn(2, 1, 1000, 1000)
    out = model(template, search)
    print("template:", template.shape)
    print("search:", search.shape)
    print("heatmap:", out.shape)