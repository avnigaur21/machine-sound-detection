# Project Context and Status

## Project goal

This project studies unsupervised anomalous sound detection (ASD): train on normal machine sounds, then identify unusual recordings that may signal a fault. The starting point is the [MERL Anomaly Score Normalization repository](https://github.com/merlresearch/anomaly-score-normalization), using the DCASE 2023 Task 2 machine-sound data and evaluator.

The comparison should use the same data splits, scoring and evaluation setup across methods. Models learn from normal training sounds. Anomaly scores on test recordings are compared with the official DCASE evaluator. No performance results have been produced yet.

## Three approaches

1. **Feature Exchange baseline** — MERL's `direct-act` representation learning method. Feature Exchange is enabled by default and can be toggled off for an ablation.
2. **Contrastive learning** — available as the `contrastive` mode; learns representations by pulling related examples together and separating other examples.
3. **Statistical contrastive learning** — available as the `stat-contrastive` mode; uses statistics from normal training audio to guide representation learning.

All three approaches are present in the project code. They have not yet been trained or compared, so we do not know which performs best. The next research phase, after establishing comparable baselines, is to inspect results and develop architecture improvements.

## Dataset and evaluator

The configured dataset is DCASE 2023. All 21 official ZIP archives are present under `DCASE2023_dataset/downloads` and were opened successfully as ZIP files. The dataset has been extracted into the expected structure:

- `DCASE2023_dataset/dev_data`: 7 machine types, each with 1,000 training and 200 test clips.
- `DCASE2023_dataset/eval_data`: 7 machine types, each with training and test clips. The training counts are 1,000 for Bandsaw, Grinder, and Shaker, and 2,000 for ToyDrone, ToyNscale, ToyTank, and Vacuum. Each has 200 test clips.
- `dcase2023_task2_evaluator`: official evaluator repository, including ground-truth labels for all 7 evaluation machine types.

The dataset is large, so preprocessing was implemented to use disk-backed/memory-mapped arrays and conservative data-loader settings to reduce RAM use.

## Implementation and safety settings

- Default method: `direct-act` with Feature Exchange enabled.
- Alternative modes: `contrastive`, `stat-contrastive`.
- Device selection: automatic CUDA, Apple MPS, or CPU selection where supported.
- Conservative starting configuration: batch size 8, one ensemble run, zero data-loader workers.
- Training-time validation is disabled to keep test/anomaly labels out of model selection.
- Results are intended to be written as timestamped CSV files.

## Current training status

A project-local `.venv314` environment was created and the required runtime packages were installed there. The environment uses CPU-only PyTorch 2.11 because no GPU is available on the current Windows machine. Preprocessing created memory-mapped cache files totaling about 19 GB.

The `direct-act` Feature Exchange run reached epoch 0, batch 687 of 1,750, then was interrupted by the user due to high system CPU and memory use. No checkpoint or metrics CSV was saved. The configuration now uses five epochs for the DCASE 2023 pilot. A smaller, sample-limited smoke run should be implemented before retrying training on this laptop.

## Reference paper

The MERL codebase is associated with **“Local Density-Based Anomaly Score Normalization for Domain Generalization.”** The paper addresses the reliability of anomaly scores under domain shifts. This project uses the MERL implementation as its foundation and extends the comparison to three representation-learning approaches.

