<div align="center">
    <h1 align="center">
        <img src="figs/icon.png" alt="Description" width="50">
        SynRS3D
    </h1>
    <h3>SynRS3D: A Synthetic Dataset for Global 3D Semantic Understanding from Monocular Remote Sensing Imagery</h3>
    <p><strong>NeurIPS 2024 Spotlight</strong></p>


[Jian Song](https://scholar.google.ch/citations?user=CgcMFJsAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Hongruixuan Chen](https://scholar.google.ch/citations?user=XOk4Cf0AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Weihao Xuan](https://weihaoxuan.com/)<sup>1,2</sup>, [Junshi Xia](https://scholar.google.com/citations?user=n1aKdTkAAAAJ&hl=en)<sup>2</sup>, [Naoto Yokoya](https://scholar.google.co.jp/citations?user=DJ2KOn8AAAAJ&hl=en)<sup>1,2</sup>

<sup>1</sup> The University of Tokyo, <sup>2</sup> RIKEN AIP

[![arXiv paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/pdf/2406.18151)


</div>

## 🛎️Updates
* **` Sep 27th, 2024`**: Codes and data are coming soon! Please stay tuned!!
* **` Sep 26th, 2024`**: SynRS3D has been accepted at NeurIPS D&B Track 2024 as a **Spotlight**!!

## Installation

1. **Clone this repository:**

    ```bash
    git clone https://github.com/JTRNEO/SynRS3D.git
    cd SynRS3D
    ```

2. **Create and activate the conda environment:**

    ```bash
    conda create -n synrs3d python=3.8
    conda activate synrs3d
    conda install pytorch=2.2.1 torchvision=0.17.1 torchaudio=2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install gdal
    pip install albumentations tqdm ever-beta==0.2.3 huggingface_hub rasterio
    ```

## Datasets Preparation (Uploading... Stay tuned.)

### Download the SynRS3D dataset:

- **SynRS3D:** [Download here](https://arxiv.org/pdf/2406.18151)

### Download the real-world datasets:

- **DFC18 (Houston dataset):** [Download here](https://arxiv.org/pdf/2406.18151)
- **DFC19 (JAX & OMA):** [Download here](https://arxiv.org/pdf/2406.18151)
- **GeoNRW (Rural & Urban):** [Download here](https://arxiv.org/pdf/2406.18151)
- **ATL:** [Download here](https://arxiv.org/pdf/2406.18151)
- **ARG:** [Download here](https://arxiv.org/pdf/2406.18151)

Place all datasets in the repository under `./SynRS3D/data`.

### Data Structure:

For SynRS3D and real-world datasets, the data is processed and stored in the following structure:

```
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/project/SynRS3D/data/DFC18
├── opt              # RGB images saved as .tif
├── gt_nDSM          # Normalized Digital Surface Model images saved as .tif
├── train.txt        # List of training data names without suffix
└── test.txt         # List of testing data names without suffix
```

For SynRS3D, it contains 17 folders. Download and extract all of them, ensuring each folder follows this structure:

```
${DATASET_ROOT} # Dataset root directory, e.g., /home/username/project/SynRS3D/data/grid_g05_mid_v1
├── opt             # RGB images saved as .tif, also post-event images in building change detection
├── pre_opt         # RGB images saved as .tif, also pre-event images in building change detection
├── gt_nDSM         # Normalized Digital Surface Model images saved as .tif
├── gt_ss_mask      # Land cover mapping labels saved as .tif
├── gt_cd_mask      # Building change detection mask saved as .tif (0 = no change, 255 = change area)
├── select_train.txt # List of training data names for the final version of SynRS3D
└── total.txt       # List of all raw SynRS3D data names without filtering
```

### Class Mapping for `gt_ss_mask` in the SynRS3D dataset:

- **Bareland:** 1
- **Grass:** 2
- **Pavement:** 3
- **Road:** 4
- **Trees:** 5
- **Water:** 6
- **Cropland:** 7
- **Buildings:** 8

## Training

You can directly use `. RS3DAda.sh` to train RS3DAda or `. sourceonly.sh` to train SynRS3D in a source-only scenario.

**Note:** If you want to evaluate land cover mapping at the same time with the OEM dataset, please download it from [this link](https://zenodo.org/records/7223446). Organize its file structure similarly to SynRS3D and enable `--eval_oem` in the script for evaluation.

## Inference Using Our Best Model

Please download the weights from [this link](https://drive.google.com/drive/folders/1sGD6KBdHuRsfAfIRlJIrs-1nfSd8cdRw?usp=sharing) and place them in the folder `./SynRS3d/pretrain`.

### Infer Height Maps

To infer height maps using trained RS3DAda on the SynRS3D dataset:

```bash
python infer_height.py \
--data_dir path/to/your/image \
--restore_from ./pretrain/RS3DAda_vitl_DPT_height.pth \
--output_path path/to/your/output \
--use_tta
```

### Infer Land Cover Maps

To infer land cover maps using trained RS3DAda on the SynRS3D dataset:

```bash
python infer_segmentation.py \
--data_dir path/to/your/image \
--restore_from ./pretrain/RS3DAda_vitl_DPT_segmentation.pth \
--output_path path/to/your/output \
--use_tta
```
