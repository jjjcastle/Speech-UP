# model.py

import torch
import torch.nn as nn

EMBED_DIM = 128

class MelEncoder(nn.Module):
    def __init__(self, emb_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):  # x: (B, 1, T, mel)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
