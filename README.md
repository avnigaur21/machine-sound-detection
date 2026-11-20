<!--
Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Anomaly Score Normalization for Domain Generalization
If you use any part of this code for your work, we kindly ask you to cite the following papers:

```
@inproceedings{wilkinghoff2025keeping,
   author = {Wilkinghoff, Kevin and Yang, Haici and Ebbers, Janek and Germain, Fran{\c{c}}ois G. and Wichern, Gordon and {Le Roux}, Jonathan},
   title = {Keeping the Balance: Anomaly Score Calculation for Domain Generalization},
   booktitle = {Proc. ICASSP},
   year = {2025},
   publisher = {IEEE}
}
```
and
```
@article{wilkinghoff2025local,
   author = {Wilkinghoff, Kevin and Yang, Haici and Ebbers, Janek and Germain, Fran{\c{c}}ois G. and Wichern, Gordon and {Le Roux}, Jonathan},
   title = {Local Density-Based Anomaly Score Normalization for Domain Generalization},
   journal = {IEEE/ACM Trans. Audio, Speech, Lang. Process.},
   year = {2025}
}
```

***

## Environment Setup

The code has been tested using `python 3.8` on both Linux and Windows, using CUDA 11.8, and requires a single GPU.
Necessary dependencies can be installed using the included `requirements.txt`:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118  # for GPU
pip install -r requirements.txt
```
***

## Datasets
Currently, [DCASE2020](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds#download), [DCASE2022](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#download), [DCASE2023](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#download), [DCASE2024](https://dcase.community/challenge2024/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#download), and [DCASE2025](https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring#download) are supported.
Each dataset should be in a folder containing the `development set`, the `additional training dataset` and the `evaluation set` with the following structure:
```
      dataset_path
      |-- dev_data
        |-- dev_machine_1
          |-- test
          |-- train
        |-- dev_machine_2
          |-- test
          |-- train
        |-- ...
      |-- eval_data
        |-- eval_machine_1
          |-- test
          |-- train
        |-- eval_machine_2
          |-- test
          |-- train
        |-- ...
```
To evaluate the performance on the evaluation set for DCASE2022, DCASE2023, DCASE2024 and DCASE2025, the ground-truth labels contained in the evaluator repositories need to be provided, i.e., [DCASE2022 evaluator](https://github.com/Kota-Dohi/dcase2022_evaluator), [DCASE2023 evaluator](https://github.com/nttcslab/dcase2023_task2_evaluator), [DCASE2024 evaluator](https://github.com/nttcslab/dcase2024_task2_evaluator), or [DCASE2025 evaluator](https://github.com/nttcslab/dcase2025_task2_evaluator). The corresponding GitHub repositories should be placed in the project folder, e.g., by using the following commands:

```bash
      git clone https://github.com/Kota-Dohi/dcase2022_evaluator.git
      git clone https://github.com/nttcslab/dcase2023_task2_evaluator.git
      git clone https://github.com/nttcslab/dcase2024_task2_evaluator.git
      git clone https://github.com/nttcslab/dcase2025_task2_evaluator.git
 ```
leading to a directory structure of:

 ```
      project_root
      |-- beats
        |-- ...
      |-- dcase2022_evaluator
        |-- ...
      |-- dcase2023_task2_evaluator
        |-- ...
      |-- dcase2024_task2_evaluator
        |-- ...
      |-- dcase2025_task2_evaluator
        |-- ...
      |-- tests
        |-- ...
      | main.py
      | conf.yaml
         .
         .
         .
   ```

   For the DCASE2020 dataset, the ground-truth labels `eval_data_list.csv` can be downloaded directly from [Zenodo](https://zenodo.org/records/3951620) and should be placed into the DCASE2020 dataset folder:
```
     DCASE2020_dataset_path
      |-- dev_data
        |-- ...
      |-- eval_data
        |-- ...
      |-- eval_data_list.csv
  ```
***

## Usage
Call the main script with the following command:
```bash
python main.py [--conf-path CONFPATH]
```

The default mode is the original MERL `direct-act` implementation. Switch representation methods and score normalization without editing the YAML file:

```bash
python main.py --mode direct-act --feature-exchange on
python main.py --mode direct-act --feature-exchange off
python main.py --mode contrastive
python main.py --mode stat-contrastive
python main.py --mode contrastive --norm-type none
python main.py --mode contrastive --norm-type LOF --norm-param 16
```

## Comparative representation experiments

The default configuration retains MERL's original `direct-act` method. In this upstream implementation, `direct-act` already enables the two-branch Feature Exchange loss. Set `model.feature_exchange_enabled: false` to run a matched ablation with the same branches and the Feature Exchange loss disabled.

Two additional modes use the same MERL reference construction, score normalization, and evaluator:

* `contrastive`: ResNet-18 log-Mel embeddings trained with NT-Xent.
* `stat-contrastive`: the same encoder trained with contrastive positives defined by k-means clusters of normal-training log-Mel statistics. The scaler and clusters are fitted from the training loader only.

These are paper-inspired implementations, not claims of exact reproduction. In particular, the contrastive paths use MERL's downstream score pipeline instead of the covariance/Mahalanobis scoring used by the DCASE 2023 contrastive report. See `conf.yaml` for the dataset, method, batch size, worker count, and device settings. DCASE 2023 is the initial shared benchmark setting.

### Memory and data handling

Preprocessing writes waveform/embedding partitions to NumPy memory-mapped cache files one sample at a time. Training and prediction read one example at a time from those caches, rather than copying all audio into RAM. The starting configuration uses batch size 8, zero DataLoader workers, and one model run. Increase these values only after a small run shows that memory and storage are sufficient. The caches still require disk space for each processed partition; no dataset is bundled or downloaded automatically.

The default training run skips validation during fitting so development test anomaly labels are not used for checkpoint selection. Final development metrics are computed after training. Keep `model.standardize: false` for the primary non-transductive comparison; enabling it estimates score statistics from test embeddings.

For the planned normalization sweep, reuse each trained checkpoint and run `model.norm_type` as `none`, `ratio`, `difference`, and `LOF`, with a fixed `norm_param` for the methods that require it. Do not compare methods with different source-reference settings.
The optional argument `--conf-path` specifies the path to the `conf.yaml` file, which contains information about the dataset to be used. This file contains all parameter values for the datasets, models, training, and embeddings that can be modified by the user. In the following, we provide more details for all parameters that are not necessarily self-explanatory:

- `model`: The embedding model being used. Based on the notation used in the paper, the following options are provided:
  - `direct-act`: an embedding model based on spectral features trained from scratch with an auxiliary classification task
  - `STFT-raw`: using the temporal mean of spectrograms without any additional training
  - `openL3-act`: an embedding model based on pre-trained openL3 embeddings trained with an auxiliary classification task
  - `openL3-raw`: an embedding model based on pre-trained openL3 embeddings without any additional training
  - `BEATs-act`: an embedding model based on pre-trained BEATs embeddings trained with an auxiliary classification task
  - `BEATs-raw`: an embedding model based on pre-trained BEATs embeddings without any additional training
  - `EAT-raw`: an embedding model based on pre-trained EAT embeddings without any additional training

  - Note: Using BEATs embeddings requires a [pre-trained BEATs model](https://onedrive.live.com/?authkey=%21ACVDQ8YOHlNK%5Frw&id=6B83B49411CA81A7%2125969&cid=6B83B49411CA81A7&parId=root&parQt=sharedby&o=OneUp) in the main folder. A list of alternative BEATs models can be found here: https://github.com/Phuriches/GenRepASD/tree/main/beats

- `norm_type`: The type of local density-based score normalization to be applied. The following options are available:
   - `None`: no normalization
   - `ratio`: the proposed normalization based on dividing by the local density estimate
   - `difference`: the proposed normalization based on subtracting the local density estimate
   - `LOF`: LOF-based normalization (requires setting a value for `K` via `norm_param`)

- `norm_param` is the hyperparameter used for the optional normalization of the anomaly scores and can be one of the following:
   - `None`: No normalization is applied.
   - `K` (int): K-NN based normalization where `K` equals the provided value is applied. Example: `16`
   - `r` (float): GWRP-based normalization where `r` equals the provided value is applied. Example: `0.9`

- `source_K_means`: Whether to apply k-means on the source domain to replace the original reference samples on the source domain. Can be either:
   - `None`: all reference samples are used
   - `K` (int): number of means for k-Means

- `standardize` (Boolean): Whether domain-wise standardization based on the test set is applied to the scores.

- `smote` (Boolean): Whether SMOTE is applied to generate additional reference samples.

- `dataset_path` should be a path to a DCASE ASD dataset, e.g., `./DCASE2023_dataset/`.

- `edition`: The edition of the DCASE ASD dataset to be used, i.e., `DCASE2020`, `DCASE2022`, `DCASE2023`, `DCASE2024`, or `DCASE2025`.

- `eval_metrics`: The user can choose between `official` and `simple` evaluation metrics when computing the performance:
  - `official`: use the normal test samples of a particular domain and all anomalous test samples of the source and target domain when computing a domain-specific AUC score. This is the same evaluation metric as used in the DCASE challenge.
  - `simple`: only use the normal and anomalous test samples belonging to the same domain when computing a domain-specific AUC score.

***
## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

***

## Copyright and License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:
```
Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

All files in `beats/`
were taken without modification from https://github.com/microsoft/unilm/tree/master/beats (license included in [LICENSES/MIT.md](LICENSES/MIT.md)):

```
Copyright (c) 2022 Microsoft
```
