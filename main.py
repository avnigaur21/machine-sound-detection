# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from argparse import ArgumentParser, Namespace
import csv
from datetime import datetime
import os
from pathlib import Path
import time

import lightning as L
import numpy as np
import torch
import yaml

from dataset import DCASEASDDataModule, find_max_size, prepare_dataset
from lightning_module import (
    ASDsystem,
    ContrastiveASDsystem,
    raw_system,
    trainedACTsystem,
)
from utils import (
    get_ASD_performance_from_scores,
    get_mean_embs,
    print_mean_and_std_performance,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--conf-path",
        default=Path("./conf.yaml"),
        type=Path,
        help="The path to the YAML configuration file (conf.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=("direct-act", "contrastive", "stat-contrastive", "STFT-raw", "openL3-act", "openL3-raw", "BEATs-act", "BEATs-raw", "EAT-raw"),
        help="Override model.mode in the YAML configuration.",
    )
    parser.add_argument(
        "--norm-type",
        choices=("none", "ratio", "difference", "LOF"),
        help="Override model.norm_type for a scoring-only normalization sweep.",
    )
    parser.add_argument(
        "--norm-param",
        help="Override model.norm_param (K for k-NN/LOF, or r for GWRP).",
    )
    parser.add_argument(
        "--feature-exchange",
        choices=("on", "off"),
        help="Enable/disable the MERL direct-act Feature Exchange loss.",
    )
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    parser.add_argument("--n-ensemble", type=int, help="Override training.n_ensemble.")
    args = parser.parse_args()

    with open(args.conf_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        hparams = Namespace(**params)
    if args.mode is not None:
        hparams.model["mode"] = args.mode
    if args.norm_type is not None:
        hparams.model["norm_type"] = args.norm_type
    if args.norm_param is not None:
        hparams.model["norm_param"] = args.norm_param
    if args.feature_exchange is not None:
        hparams.model["feature_exchange_enabled"] = args.feature_exchange == "on"
    if args.batch_size is not None:
        hparams.training["batch_size"] = args.batch_size
    if args.n_ensemble is not None:
        hparams.training["n_ensemble"] = args.n_ensemble
    if hparams.training["batch_size"] < 1 or hparams.training["n_ensemble"] < 1:
        raise ValueError("batch_size and n_ensemble must both be positive.")

    dataset_path = hparams.dataset["path"]
    edition = hparams.dataset["edition"]
    mode = hparams.model["mode"]
    source_K = str(hparams.model["source_K_means"])
    if source_K.lower() == "none":
        source_K = 0
    else:
        source_K = int(source_K)
    norm_param = str(hparams.model["norm_param"])
    if norm_param.lower() == "none":
        norm_param = None
    norm_type = str(hparams.model["norm_type"])
    if norm_type.lower() == "none":
        norm_type = None
    if not (
        norm_type is None
        or norm_type == "ratio"
        or norm_type == "difference"
        or norm_type == "LOF"
    ):
        raise TypeError(
            'only supported normalization types are "ratio", "difference", or "LOF"!'
        )
    standardize = hparams.model["standardize"]
    smote = hparams.model["smote"]
    smote_num_neighbors = int(str(hparams.model["smote_num_neighbors"]))

    # check for optional normalization parameter
    K = None
    r = None
    if norm_param is not None:
        if norm_param.isdigit():
            K = int(norm_param)
        else:
            try:
                r = float(norm_param)
            except ValueError:
                raise TypeError(
                    'The norm parameter must be either "None", int, or float!'
                )
    if K is None and norm_type == "LOF":
        raise TypeError("LOF requires a value of K to be provided!")

    # prepare data
    if mode in ("direct-act", "contrastive", "stat-contrastive"):
        target_dir = "./" + edition + "_processed"
        prepare_dataset(
            data_dir=dataset_path, target_dir=target_dir, edition=edition
        )
        input_dim = find_max_size(data_dir=dataset_path)
        if mode == "direct-act":
            max_epochs = hparams.training["num_epochs_direct-act"]
        elif mode == "contrastive":
            max_epochs = hparams.training["num_epochs_contrastive"]
        else:
            max_epochs = hparams.training["num_epochs_stat-contrastive"]
    elif mode == "openL3-act" or mode == "openL3-raw":
        target_dir = "./" + edition + "_openL3_processed"
        prepare_dataset(
            data_dir=dataset_path,
            target_dir=target_dir,
            edition=edition,
            new_sr=48000,  # required for openL3, does not need to be modified
            pre_model="openL3",
        )
        input_dim = hparams.embs["openL3_input_dim"]
        if mode == "openL3-act":
            max_epochs = hparams.training["num_epochs_openL3-act"]
        else:
            max_epochs = 0  # no training for direct usage
    elif mode == "BEATs-act" or mode == "BEATs-raw":
        target_dir = "./" + edition + "_BEATs_processed"
        prepare_dataset(
            data_dir=dataset_path,
            target_dir=target_dir,
            edition=edition,
            pre_model="BEATs",
        )
        input_dim = hparams.embs["BEATs_input_dim"]
        if mode == "BEATs-act":
            max_epochs = hparams.training["num_epochs_BEATs-act"]
        else:
            max_epochs = 0  # no training for direct usage
    elif mode == "EAT-raw":
        target_dir = "./" + edition + "_EAT_processed"
        prepare_dataset(
            data_dir=dataset_path,
            target_dir=target_dir,
            edition=edition,
            pre_model="EAT",
        )
        input_dim = hparams.embs["EAT_input_dim"]
        max_epochs = 0  # no training for direct usage
    elif mode == "STFT-raw":
        target_dir = "./" + edition + "_STFT_processed"
        prepare_dataset(
            data_dir=dataset_path,
            target_dir=target_dir,
            edition=edition,
            pre_model="STFT",
        )
        input_dim = int(find_max_size(data_dir=dataset_path) / 2)
        max_epochs = 0  # no training for direct usage
    else:
        raise TypeError(
            'only supported modes are "direct-act", "STFT-raw", "openL3-act", "openL3-raw", "BEATs-act", "BEATs-raw", or "EAT-raw"!'
        )
    dcase_asd = DCASEASDDataModule(
        data_dir=target_dir,
        batch_size=hparams.training["batch_size"],
        num_workers=hparams.training.get("num_workers", 0),
        use_eval_for_train=True,
        use_dev_for_train=True,
        use_dev_for_pred=True,
        use_eval_for_pred=False,
    )
    if (
        mode == "openL3-raw"
        or mode == "BEATs-raw"
        or mode == "EAT-raw"
        or mode == "STFT-raw"
    ):
        dcase_asd.setup("fit")

    n_ensemble = hparams.training["n_ensemble"]
    if edition == "DCASE2020":
        n_metrics = 6
    else:
        n_metrics = 9
    all_scores_eval = []
    runtimes_seconds = []
    performances_eval = np.zeros((n_ensemble, n_metrics))
    all_scores_dev = []
    performances_dev = np.zeros((n_ensemble, n_metrics))
    for k_ensemble in np.arange(n_ensemble):
        run_started_at = time.perf_counter()
        L.seed_everything(
            int(hparams.training.get("seed", 42)) + int(k_ensemble),
            workers=True,
        )
        print("Running ensemble iteration #" + str(k_ensemble + 1))
        tb_logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir="logs/",
            name=edition + "_logs" + "_k_ensemble_" + str(k_ensemble),
        )

        # setup model
        if mode == "direct-act":
            asd_system = ASDsystem(
                input_dim=input_dim,
                num_classes=dcase_asd.num_classes,
                subspace_dim=hparams.model["subspace_dim"],
                bias=hparams.model["bias"],
                affine=hparams.model["affine"],
                trainable_centers=hparams.model["trainable_centers"],
                feature_exchange_enabled=hparams.model.get(
                    "feature_exchange_enabled", True
                ),
                source_K=source_K,
                K=K,
                r=r,
                norm_type=norm_type,
                standardize=standardize,
                smote=smote,
                smote_num_neighbors=smote_num_neighbors,
            )
        elif mode in ("contrastive", "stat-contrastive"):
            asd_system = ContrastiveASDsystem(
                method=mode,
                n_clusters=hparams.model.get("n_clusters", 8),
                temperature=hparams.model.get("temperature", 0.05),
                source_K=source_K,
                K=K,
                r=r,
                norm_type=norm_type,
                standardize=standardize,
                smote=smote,
                smote_num_neighbors=smote_num_neighbors,
            )
        elif mode == "openL3-act" or mode == "BEATs-act":
            asd_system = trainedACTsystem(
                input_dim=input_dim,
                num_classes=dcase_asd.num_classes,
                subspace_dim=hparams.model["subspace_dim"],
                bias=hparams.model["bias"],
                affine=hparams.model["affine"],
                trainable_centers=hparams.model["trainable_centers"],
                source_K=source_K,
                K=K,
                r=r,
                norm_type=norm_type,
                standardize=standardize,
                smote=smote,
                smote_num_neighbors=smote_num_neighbors,
            )
        elif (
            mode == "openL3-raw"
            or mode == "BEATs-raw"
            or mode == "EAT-raw"
            or mode == "STFT-raw"
        ):
            asd_system = raw_system(
                source_K=source_K,
                K=K,
                r=r,
                norm_type=norm_type,
                standardize=standardize,
                smote=smote,
                smote_num_neighbors=smote_num_neighbors,
            )
        accelerator = hparams.training.get("accelerator", "auto")
        trainer = L.Trainer(
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
            logger=tb_logger,
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=False,
            limit_val_batches=(
                1.0
                if hparams.training.get("validate_during_training", False)
                else 0
            ),
        )  # no sanity checks because means are not yet available

        # train model
        if mode in (
            "direct-act",
            "contrastive",
            "stat-contrastive",
            "openL3-act",
            "BEATs-act",
        ):
            model_name = "#".join(
                [
                    edition,
                    mode,
                    str(k_ensemble),
                    str(max_epochs),
                    str(hparams.model["subspace_dim"]),
                    str(hparams.model["bias"]),
                    str(hparams.model["affine"]),
                    str(hparams.model["trainable_centers"]),
                    str(hparams.model.get("feature_exchange_enabled", True)),
                    str(hparams.model.get("n_clusters", 8)),
                    str(hparams.model.get("temperature", 0.05)),
                    ".ckpt",
                ]
            )
            model_dir = Path(hparams.training["model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / model_name
            if model_path.is_file() and hparams.training["use_checkpoints"]:
                if mode == "direct-act":
                    asd_system = ASDsystem.load_from_checkpoint(
                        checkpoint_path=model_path
                    )
                elif mode in ("contrastive", "stat-contrastive"):
                    asd_system = ContrastiveASDsystem.load_from_checkpoint(
                        checkpoint_path=model_path
                    )
                elif mode == "openL3-act" or mode == "BEATs-act":
                    asd_system = trainedACTsystem.load_from_checkpoint(
                        checkpoint_path=model_path
                    )
                asd_system.source_K = source_K
                asd_system.K = K
                asd_system.r = r
                asd_system.norm_type = norm_type
                asd_system.standardize = standardize
                asd_system.smote = smote
                asd_system.smote_num_neighbors = smote_num_neighbors
                dcase_asd.setup("fit")
            else:
                trainer.fit(asd_system, dcase_asd)
                if hparams.training["use_checkpoints"]:
                    trainer.save_checkpoint(model_path)

        # reload all data
        asd_system.eval()  # disables randomness
        asd_system.to(trainer.strategy.root_device)
        # re-calculate means
        for batch_idx, batch in enumerate(dcase_asd.train_dataloader()):
            asd_system.reload_training_embs(batch, batch_idx)
        (
            asd_system.means,
            asd_system.mean_machine_ids,
            asd_system.mean_source_labels,
        ) = get_mean_embs(
            asd_system.train_embs,
            asd_system.train_machine_ids,
            asd_system.train_source_labels,
            k=asd_system.source_K,
            smote=asd_system.smote,
            smote_num_neighbors=asd_system.smote_num_neighbors,
            use_mse=asd_system.use_mse,
        )

        # evaluate performance on dev data
        dcase_asd_test = DCASEASDDataModule(
            data_dir=target_dir,
            batch_size=hparams.training["batch_size"],
            num_workers=hparams.training.get("num_workers", 0),
            use_eval_for_train=False,
            use_eval_for_pred=False,
            use_dev_for_train=True,
            use_dev_for_pred=True,
        )
        if standardize:
            asd_system.print_results = False
            trainer.validate(
                asd_system, dcase_asd_test, verbose=False
            )  # used to calculate the statistics

        # predict with trained model and collect scores
        preds_dev = trainer.predict(asd_system, dcase_asd_test)
        scores_dev, machine_ids_dev, source_labels_dev, anomaly_labels_dev = (
            tuple(map(torch.cat, zip(*preds_dev)))
        )
        all_scores_dev.append(scores_dev)

        # compute performance for current model only
        performances_dev[k_ensemble] = np.array(
            get_ASD_performance_from_scores(
                scores_dev,
                machine_ids_dev,
                source_labels_dev,
                anomaly_labels_dev,
                print_results=True,
                return_domain_specific=edition != "DCASE2020",
                eval_metrics=hparams.dataset["eval_metrics"],
            )
        )

        # evaluate performance on evaluation data
        dcase_asd_test = DCASEASDDataModule(
            data_dir=target_dir,
            batch_size=hparams.training["batch_size"],
            num_workers=hparams.training.get("num_workers", 0),
            use_eval_for_train=True,
            use_eval_for_pred=True,
            use_dev_for_train=False,
            use_dev_for_pred=False,
        )
        if standardize:
            asd_system.print_results = False
            trainer.test(
                asd_system, dcase_asd_test
            )  # used to calculate the statistics

        # predict with trained model and collect scores
        preds_eval = trainer.predict(asd_system, dcase_asd_test)
        (
            scores_eval,
            machine_ids_eval,
            source_labels_eval,
            anomaly_labels_eval,
        ) = tuple(map(torch.cat, zip(*preds_eval)))
        all_scores_eval.append(scores_eval)

        # compute performance for current model only
        performances_eval[k_ensemble] = np.array(
            get_ASD_performance_from_scores(
                scores_eval,
                machine_ids_eval,
                source_labels_eval,
                anomaly_labels_eval,
                print_results=False,
                return_domain_specific=edition != "DCASE2020",
                eval_metrics=hparams.dataset["eval_metrics"],
            )
        )
        asd_system.on_train_epoch_end()  # delete stored training embeddings
        runtimes_seconds.append(time.perf_counter() - run_started_at)

    # output performance
    print(
        "################################################################################"
    )
    print("Mean and standard deviation of results on development set")
    print(
        "################################################################################"
    )
    print_mean_and_std_performance(
        performances_dev, return_domain_specific=edition != "DCASE2020"
    )
    print(
        "################################################################################"
    )
    print("Performance of ensemble on development set")
    print(
        "################################################################################"
    )
    get_ASD_performance_from_scores(
        torch.stack(all_scores_dev).mean(dim=0),
        machine_ids_dev,
        source_labels_dev,
        anomaly_labels_dev,
        print_results=True,
        return_domain_specific=edition != "DCASE2020",
        eval_metrics=hparams.dataset["eval_metrics"],
    )

    metric_names = (
        [
            "mean_auc",
            "mean_pauc",
            "mean_hmean_auc_pauc",
            "hmean_auc",
            "hmean_pauc",
            "hmean_all",
        ]
        if edition == "DCASE2020"
        else [
            f"{domain}_{metric}"
            for domain in ("source", "target", "mixed")
            for metric in ("hmean_auc", "hmean_pauc", "hmean_all")
        ]
    )
    norm_label = norm_type or "none"
    result_dir = Path(hparams.training.get("results_dir", "results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / (
        f"{edition}_{mode}_{norm_label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
    )
    with result_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=("edition", "method", "norm_type", "seed", "split", "runtime_seconds")
            + tuple(metric_names),
        )
        writer.writeheader()
        for split_name, performances in (
            ("development", performances_dev),
            ("evaluation", performances_eval),
        ):
            for run_index, values in enumerate(performances):
                row = {
                    "edition": edition,
                    "method": mode,
                    "norm_type": norm_label,
                    "seed": int(hparams.training.get("seed", 42)) + run_index,
                    "split": split_name,
                    "runtime_seconds": runtimes_seconds[run_index],
                }
                row.update(dict(zip(metric_names, values.tolist())))
                writer.writerow(row)
    print(f"Per-run metrics written to {result_path}")
    print(
        "################################################################################"
    )
    print("Mean and standard deviation of results on evaluation set")
    print(
        "################################################################################"
    )
    print_mean_and_std_performance(
        performances_eval, return_domain_specific=edition != "DCASE2020"
    )
    print(
        "################################################################################"
    )
    print("Performance of ensemble on evaluation set")
    print(
        "################################################################################"
    )
    get_ASD_performance_from_scores(
        torch.stack(all_scores_eval).mean(dim=0),
        machine_ids_eval,
        source_labels_eval,
        anomaly_labels_eval,
        print_results=True,
        return_domain_specific=edition != "DCASE2020",
        eval_metrics=hparams.dataset["eval_metrics"],
    )
