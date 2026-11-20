# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import abstractmethod

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from network import (
    AdaProj,
    FFT_branch,
    STFT_branch,
    STFT_extractor,
    categorical_cross_entropy,
    feature_exchange,
    mixup_data,
    shallow_classifier,
)
from utils import (
    get_ASD_performance,
    get_mean_embs,
    get_scores,
    get_standardization_stats,
)
from representation_methods import (
    ContrastiveAudioEncoder,
    apply_mel_time_mask,
    augment_waveforms,
    nt_xent_loss,
    statistical_features,
)


class base_emb_model(L.LightningModule):
    def __init__(
        self,
        source_K: int = None,
        K: int = None,
        r: float = None,
        norm_type: str = "ratio",
        standardize: bool = False,
        smote: bool = False,
        smote_num_neighbors: int = 2,
        print_results: bool = True,
    ):
        super().__init__()
        self.use_mse = False
        self.K = K
        self.r = r
        self.source_K = source_K
        self.norm_type = norm_type
        self.standardize = standardize
        self.smote = smote
        self.smote_num_neighbors = smote_num_neighbors
        self.print_results = print_results
        self.save_hyperparameters()

        # tensors needed for evaluation
        self.train_embs = []
        self.train_machine_ids = []
        self.train_source_labels = []
        self.means = []
        self.mean_machine_ids = []
        self.mean_source_labels = []
        self.val_embs = []
        self.val_machine_ids = []
        self.val_source_labels = []
        self.val_anomaly_labels = []
        self.test_embs = []
        self.test_machine_ids = []
        self.test_source_labels = []
        self.test_anomaly_labels = []
        self.dists_source_mean = []
        self.dists_source_std = []
        self.dists_target_mean = []
        self.dists_target_std = []

    @abstractmethod
    def forward(self, x):
        pass

    def reload_training_embs(self, batch, batch_nb):
        x, y, machine_id, source = batch
        x = x.to(self.device)
        machine_id = machine_id.to(self.device)
        source = source.to(self.device)
        embs_orig = self.get_normalized_embs_(x)
        self.train_machine_ids.append(machine_id.detach())
        self.train_embs.append(embs_orig.detach())
        self.train_source_labels.append(source.detach())
        return

    @abstractmethod
    def training_step(self, batch, batch_nb):
        pass

    @abstractmethod
    def get_normalized_embs_(self, x):
        pass

    def validation_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        embs = self.get_normalized_embs_(x)
        self.val_embs.append(embs.detach())
        self.val_machine_ids.append(machine_id.detach())
        self.val_source_labels.append(source.detach())
        self.val_anomaly_labels.append(y.detach())
        return

    def test_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        embs = self.get_normalized_embs_(x)
        self.test_embs.append(embs.detach())
        self.test_machine_ids.append(machine_id.detach())
        self.test_source_labels.append(source.detach())
        self.test_anomaly_labels.append(y.detach())
        return

    def predict_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        embs = self.get_normalized_embs_(x)
        scores = get_scores(
            self.means,
            self.mean_machine_ids,
            self.mean_source_labels,
            embs,
            machine_id,
            use_mse=self.use_mse,
            K=self.K,
            r=self.r,
            norm_type=self.norm_type,
            standardize=self.standardize,
            dists_source_mean=self.dists_source_mean,
            dists_source_std=self.dists_source_std,
            dists_target_mean=self.dists_target_mean,
            dists_target_std=self.dists_target_std,
        )
        return scores, machine_id, source, y

    def on_validation_epoch_end(self):
        # apply k-means on all training samples
        self.means, self.mean_machine_ids, self.mean_source_labels = (
            get_mean_embs(
                self.train_embs,
                self.train_machine_ids,
                self.train_source_labels,
                k=self.source_K,
                smote=self.smote,
                smote_num_neighbors=self.smote_num_neighbors,
                use_mse=self.use_mse,
            )
        )

        if self.standardize:
            (
                self.dists_source_mean,
                self.dists_source_std,
                self.dists_target_mean,
                self.dists_target_std,
            ) = get_standardization_stats(
                self.means,
                self.mean_machine_ids,
                self.mean_source_labels,
                self.val_embs,
                self.val_machine_ids,
                use_mse=self.use_mse,
                K=self.K,
                r=self.r,
                norm_type=self.norm_type,
            )

        # compute performance
        aauc, apauc, amauc, hauc, hpauc, hmauc = get_ASD_performance(
            self.means,
            self.mean_machine_ids,
            self.mean_source_labels,
            self.val_embs,
            self.val_machine_ids,
            self.val_source_labels,
            self.val_anomaly_labels,
            print_results=self.print_results,
            use_mse=self.use_mse,
            K=self.K,
            r=self.r,
            norm_type=self.norm_type,
            standardize=self.standardize,
            dists_source_mean=self.dists_source_mean,
            dists_source_std=self.dists_source_std,
            dists_target_mean=self.dists_target_mean,
            dists_target_std=self.dists_target_std,
        )
        log_values = {
            "amean_val_AUC": aauc,
            "amean_val_pAUC": apauc,
            "amean_val_mAUC": amauc,
            "hmean_val_AUC": hauc,
            "hmean_val_pAUC": hpauc,
            "hmean_val_mAUC": hmauc,
        }
        self.log_dict(log_values, prog_bar=True, on_epoch=True, on_step=False)

        # free memory
        self.val_embs.clear()
        self.val_machine_ids.clear()
        self.val_source_labels.clear()
        self.val_anomaly_labels.clear()
        return

    def on_test_epoch_end(self):
        if self.standardize:
            (
                self.dists_source_mean,
                self.dists_source_std,
                self.dists_target_mean,
                self.dists_target_std,
            ) = get_standardization_stats(
                self.means,
                self.mean_machine_ids,
                self.mean_source_labels,
                self.test_embs,
                self.test_machine_ids,
                use_mse=self.use_mse,
                K=self.K,
                r=self.r,
                norm_type=self.norm_type,
            )

        # compute performance
        aauc, apauc, amauc, hauc, hpauc, hmauc = get_ASD_performance(
            self.means,
            self.mean_machine_ids,
            self.mean_source_labels,
            self.test_embs,
            self.test_machine_ids,
            self.test_source_labels,
            self.test_anomaly_labels,
            print_results=self.print_results,
            use_mse=self.use_mse,
            K=self.K,
            r=self.r,
            norm_type=self.norm_type,
            standardize=self.standardize,
            dists_source_mean=self.dists_source_mean,
            dists_source_std=self.dists_source_std,
            dists_target_mean=self.dists_target_mean,
            dists_target_std=self.dists_target_std,
        )

        # free memory
        self.test_embs.clear()
        self.test_machine_ids.clear()
        self.test_source_labels.clear()
        self.test_anomaly_labels.clear()
        return

    def on_train_epoch_end(self):
        # free memory
        self.train_embs.clear()
        self.train_machine_ids.clear()
        self.train_source_labels.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=1e-5, eps=1e-7
        )  # adjust settings to keras default


class raw_system(base_emb_model):
    def __init__(
        self,
        source_K: int = None,
        K: int = None,
        r: float = None,
        norm_type: str = "ratio",
        smote: bool = False,
        smote_num_neighbors: int = 2,
        standardize: bool = False,
    ):
        super().__init__(
            source_K=source_K,
            K=K,
            r=r,
            norm_type=norm_type,
            standardize=standardize,
            smote=smote,
            smote_num_neighbors=smote_num_neighbors,
        )
        self.use_mse = True

    def forward(self, x):
        return x

    def training_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        embs_orig = self(x)
        self.train_machine_ids.append(machine_id.detach())
        self.train_embs.append(embs_orig.detach())
        self.train_source_labels.append(source.detach())
        return None

    def get_normalized_embs_(self, x):
        embs = self(x)  # no normalization for raw embeddings
        return embs


class ContrastiveASDsystem(base_emb_model):
    """Shared MERL scoring with NT-Xent or statistical-cluster supervision."""

    def __init__(
        self,
        method: str = "contrastive",
        n_clusters: int = 8,
        temperature: float = 0.05,
        source_K: int = None,
        K: int = None,
        r: float = None,
        norm_type: str = "ratio",
        standardize: bool = False,
        smote: bool = False,
        smote_num_neighbors: int = 2,
    ):
        super().__init__(
            source_K=source_K,
            K=K,
            r=r,
            norm_type=norm_type,
            standardize=standardize,
            smote=smote,
            smote_num_neighbors=smote_num_neighbors,
        )
        if method not in ("contrastive", "stat-contrastive"):
            raise ValueError(f"Unknown representation method: {method}")
        self.method = method
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.encoder = ContrastiveAudioEncoder()
        self.cluster_scaler = None
        self.cluster_model = None

    def forward(self, waveforms):
        return self.encoder(waveforms)[0]

    def get_normalized_embs_(self, waveforms):
        return F.normalize(self(waveforms), p=2.0, dim=1)

    def on_fit_start(self):
        if self.method != "stat-contrastive":
            return
        # Fit the pseudo-label model from normal training audio only.
        feature_batches = []
        for batch in self.trainer.datamodule.train_dataloader():
            features = statistical_features(self.encoder, batch[0].to(self.device))
            feature_batches.append(features.detach().cpu().numpy())
        if not feature_batches:
            raise RuntimeError("No normal training audio available for clustering.")
        features = np.concatenate(feature_batches, axis=0)
        self.cluster_scaler = StandardScaler().fit(features)
        scaled = self.cluster_scaler.transform(features)
        cluster_count = min(self.n_clusters, len(scaled))
        if cluster_count < 2:
            raise RuntimeError("Statistical clustering requires at least two training recordings.")
        self.cluster_model = KMeans(
            n_clusters=cluster_count, random_state=0, n_init=10
        ).fit(scaled)

    def training_step(self, batch, batch_nb):
        waveforms = batch[0]
        view_a = augment_waveforms(waveforms)
        view_b = augment_waveforms(waveforms)
        mel_a = apply_mel_time_mask(self.encoder.log_mel(view_a))
        mel_b = apply_mel_time_mask(self.encoder.log_mel(view_b))
        emb_a = self.encoder.backbone(mel_a.unsqueeze(1))
        emb_b = self.encoder.backbone(mel_b.unsqueeze(1))
        z_a = F.normalize(self.encoder.projector(emb_a), dim=1)
        z_b = F.normalize(self.encoder.projector(emb_b), dim=1)

        cluster_labels = None
        if self.method == "stat-contrastive":
            features = statistical_features(self.encoder, waveforms)
            labels = self.cluster_model.predict(
                self.cluster_scaler.transform(features.detach().cpu().numpy())
            )
            cluster_labels = torch.as_tensor(labels, device=self.device)
        loss = nt_xent_loss(
            z_a, z_b, temperature=self.temperature, labels=cluster_labels
        )
        embs = self.get_normalized_embs_(waveforms)
        self.train_machine_ids.append(batch[2].detach())
        self.train_embs.append(embs.detach())
        self.train_source_labels.append(batch[3].detach())
        self.log("train_contrastive_loss", loss, on_epoch=True, prog_bar=True)
        return loss


class trainedACTsystem(base_emb_model):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        subspace_dim: int = 16,
        bias: bool = True,
        affine: bool = True,
        trainable_centers: bool = False,
        source_K: int = None,
        K: int = None,
        r: float = None,
        norm_type: str = "ratio",
        standardize: bool = False,
        smote: bool = False,
        smote_num_neighbors: int = 2,
    ):
        super().__init__(
            source_K=source_K,
            K=K,
            r=r,
            norm_type=norm_type,
            standardize=standardize,
        )
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.subspace_dim = subspace_dim
        self.bias = bias
        self.affine = affine
        self.trainable_centers = trainable_centers

        # network modules
        self.emb_net = shallow_classifier(
            bias=self.bias,
            affine=self.affine,
            emb_dim=128,
            input_dim=self.input_dim,
        )
        self.adaproj = AdaProj(
            emb_dim=128,
            num_classes=self.num_classes,
            subspace_dim=self.subspace_dim,
            trainable=self.trainable_centers,
        )

    def forward(self, x):
        return self.emb_net(x)

    def training_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        # data augmentation
        x_mix, y_mix = mixup_data(x, y, p=1)
        # compute loss and accuracy
        embs_orig = self(x)
        embs_mix = self(x_mix)
        logits = self.adaproj(embs_mix, y_mix)
        loss = categorical_cross_entropy(logits, y_mix)
        embs_orig = F.normalize(embs_orig, p=2.0, dim=1)  # normalize
        self.train_machine_ids.append(machine_id.detach())
        self.train_embs.append(embs_orig.detach())
        self.train_source_labels.append(source.detach())
        log_values = {"loss": loss}
        self.log_dict(log_values, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def get_normalized_embs_(self, x):
        embs = self(x)
        embs = F.normalize(embs, p=2.0, dim=1)  # normalize
        return embs


class ASDsystem(base_emb_model):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        subspace_dim: int = 16,
        bias: bool = True,
        affine: bool = True,
        trainable_centers: bool = False,
        nfft: int = 1024,
        use_fft: bool = True,
        feature_exchange_enabled: bool = True,
        temporal_normalization: bool = True,
        source_K: int = None,
        K: int = None,
        r: float = None,
        norm_type: str = "ratio",
        standardize: bool = False,
        smote: bool = False,
        smote_num_neighbors: int = 2,
    ):
        super().__init__(
            source_K=source_K,
            K=K,
            r=r,
            norm_type=norm_type,
            standardize=standardize,
            smote=smote,
            smote_num_neighbors=smote_num_neighbors,
        )
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.subspace_dim = subspace_dim
        self.bias = bias
        self.affine = affine
        self.trainable_centers = trainable_centers
        self.nfft = nfft
        self.use_fft = use_fft
        self.feature_exchange_enabled = feature_exchange_enabled
        self.temporal_normalization = temporal_normalization
        self.stft = STFT_extractor(
            affine=self.affine,
            nfft=self.nfft,
            temporal_normalization=self.temporal_normalization,
        )
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )

        # network modules
        self.stft_branch = STFT_branch(bias=self.bias, affine=self.affine)
        if self.use_fft:
            self.fft_branch = FFT_branch(
                input_dim=self.input_dim, bias=self.bias, affine=self.affine
            )
            self.adaproj_featex = AdaProj(
                emb_dim=512,
                num_classes=3 * self.num_classes,
                subspace_dim=self.subspace_dim,
                trainable=True,
            )
            self.adaproj = AdaProj(
                emb_dim=512,
                num_classes=self.num_classes,
                subspace_dim=self.subspace_dim,
                trainable=self.trainable_centers,
            )
        else:
            self.adaproj = AdaProj(
                emb_dim=256,
                num_classes=self.num_classes,
                subspace_dim=self.subspace_dim,
                trainable=self.trainable_centers,
            )

    def forward(self, x):
        if len(x.size()) == 2:
            emb_stft = self.stft_branch(self.stft(x))
        else:
            emb_stft = self.stft_branch(x)
        if self.use_fft:
            emb_fft = self.fft_branch(x)
            return emb_fft, emb_stft
        else:
            return emb_stft

    def training_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        # data augmentation
        x_mix, y_mix = mixup_data(x, y, p=0.5)
        # compute loss and accuracy
        if self.use_fft:
            embs_fft, embs_stft = self(x)
            embs_orig = torch.cat((embs_fft, embs_stft), dim=1)
            embs_fft_mix, embs_stft_mix = self(x_mix)
            embs_mix = torch.cat((embs_fft_mix, embs_stft_mix), dim=1)
            logits = self.adaproj(embs_mix, y_mix)
            loss = categorical_cross_entropy(logits, y_mix)
            if self.feature_exchange_enabled:
                embs_featex, y_featex = feature_exchange(
                    embs_fft_mix, embs_stft_mix, y_mix, p=0.5
                )
                logits_featex = self.adaproj_featex(embs_featex, y_featex)
                loss = loss + categorical_cross_entropy(logits_featex, y_featex)
        else:
            embs_orig = self(x)
            embs_mix = self(x_mix)
            logits = self.adaproj(embs_mix, y_mix)
            loss = categorical_cross_entropy(logits, y_mix)
        embs_orig = F.normalize(embs_orig, p=2.0, dim=1)  # normalize
        self.train_machine_ids.append(machine_id.detach())
        self.train_embs.append(embs_orig.detach())
        self.train_source_labels.append(source.detach())
        log_values = {"loss": loss}
        self.log_dict(log_values, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def get_normalized_embs_(self, x):
        if self.use_fft:
            embs_fft, embs_stft = self(x)
            embs = torch.cat((embs_fft, embs_stft), dim=1)
        else:
            embs = self(x)
        embs = F.normalize(embs, p=2.0, dim=1)  # normalize
        return embs
