# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audio representation modules used by the comparative ASD experiments."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision.models import resnet18


class ContrastiveAudioEncoder(nn.Module):
    """ResNet-18 Mel encoder with a separate contrastive projection head."""

    def __init__(self, embedding_dim: int = 128, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            power=2.0,
        )
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def log_mel(self, waveforms):
        return self.mel(waveforms).clamp_min(1e-6).log()

    def forward(self, waveforms):
        features = self.log_mel(waveforms).unsqueeze(1)
        embedding = self.backbone(features)
        projection = F.normalize(self.projector(embedding), dim=1)
        return embedding, projection


def augment_waveforms(waveforms, sample_rate=16000, time_mask=True):
    """Apply lightweight audio-preserving views for contrastive training."""
    views = []
    for waveform in waveforms:
        view = waveform.detach().cpu().clone()
        if random.random() < 0.5:
            steps = random.randint(-12, 12)
            if steps:
                view = torchaudio.functional.pitch_shift(
                    view, sample_rate=sample_rate, n_steps=steps
                )
        if random.random() < 0.5:
            signal_rms = view.square().mean().sqrt().clamp_min(1e-8)
            snr_db = random.uniform(6.0, 24.0)
            noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
            view = view + torch.randn_like(view) * noise_rms
        views.append(view)
    batch = torch.stack(views).to(waveforms.device)
    return batch


def apply_mel_time_mask(log_mel, sample_rate=16000, hop_length=512):
    """Mask 0.5 or 1.0 seconds on each log-Mel example."""
    masked = log_mel.clone()
    frames = masked.shape[-1]
    for row in masked:
        duration = random.choice((0.5, 1.0))
        width = min(max(1, int(duration * sample_rate / hop_length)), frames)
        start = random.randint(0, max(0, frames - width))
        row[..., start : start + width] = 0
    return masked


def nt_xent_loss(view_a, view_b, temperature=0.05, labels=None):
    """NT-Xent; optional pseudo-labels turn it into supervised contrastive loss."""
    z = F.normalize(torch.cat((view_a, view_b), dim=0), dim=1)
    logits = torch.mm(z, z.T) / temperature
    batch = view_a.shape[0]
    self_mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_denominator = torch.logsumexp(logits, dim=1)

    if labels is None:
        positives = torch.cat(
            (
                torch.arange(batch, 2 * batch, device=z.device),
                torch.arange(batch, device=z.device),
            )
        )
        log_probability = logits[
            torch.arange(2 * batch, device=z.device), positives
        ] - log_denominator
        return -log_probability.mean()

    labels = labels.to(z.device).reshape(-1)
    labels = torch.cat((labels, labels), dim=0)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    log_probabilities = logits - log_denominator[:, None]
    positive_log_probabilities = log_probabilities.masked_fill(~positive_mask, 0)
    return -(positive_log_probabilities.sum(dim=1) / positive_count).mean()


def statistical_features(encoder, waveforms):
    """Compact temporal mean/std summaries used to form training-only clusters."""
    with torch.no_grad():
        log_mel = encoder.log_mel(waveforms)
        return torch.cat(
            (log_mel.mean(dim=-1), log_mel.std(dim=-1, unbiased=False)), dim=1
        )
