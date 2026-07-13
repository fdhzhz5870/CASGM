# CASGM

CASGM: Co-Visible Area Guided Optical-SAR Matching with Significant Geometric Differences

> **📌 Dataset Availability**
> - **GeoMS-OS Test Dataset (this directory):** Publicly available. The download link is provided in the [Dataset Download](#dataset-download) section below.
> - **CASGM** and the remaining **GeoMS-OS datasets** (e.g., training set and the complete dataset): These will be released after the paper is officially published. Please stay tuned.

---

# GeoMS-OS Test Dataset

This directory contains paired optical and SAR test data organized by scale and dataset subset. Each sample consists of an optical image, a SAR image, and a homography matrix that maps coordinates from the optical image to the SAR image.

## Dataset Download

- **GeoMS-OS Test Dataset (Public)**  
  Download: https://huggingface.co/datasets/PuYuchi/GeoMS-OS

  The download includes the public test subsets for both **scale12** and **scale24**.

## Directory Structure

```text
GeoMS-OS_test/
+-- scale12/
|   +-- 0_5m/
|   |   +-- test/
|   |       +-- H/
|   |       +-- opt/
|   |       +-- sar/
|   +-- 1m/
|   +-- 5m/
|   +-- 10m/
|   +-- MapData/
|   +-- OSdataset/
|   +-- WHU-SEN-City/
+-- scale24/
    +-- 0_5m/
    +-- 1m/
    +-- 5m/
    +-- 10m/
    +-- MapData/
    +-- OSdataset/
    +-- WHU-SEN-City/
```

# Optical–SAR Homography Data Generator

This repository generates paired optical and SAR image crops with synthetic homographies for image-registration training and evaluation.

## Input layout

Each input dataset must contain matching filenames in two directories:

```text
input_dataset/
├── optical/
│   └── example.png
└── sar/
    └── example.png
```

Supported source extensions are TIFF, PNG, JPEG, and BMP. A pair is used only when the optical and SAR filenames are identical.

## Output layout

```text
output_dataset/
├── opt/    # Generated optical crops
├── sar/    # Generated SAR crops
└── H/      # One flattened 3 x 3 homography per crop pair
```

Each homography file contains nine space-separated values. Files in `opt`, `sar`, and `H` share the same stem.

## Installation

Python 3.9 or later is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your operating system and CUDA version if the default package does not match your environment.

## Usage

Adjust augmentation settings in `gen_data.yaml`, then run:

```bash
python gen_main.py INPUT_DATASET OUTPUT_DATASET --num-sample 5 --cropnum 4 --max-pairs 80000 --stype 0
```

Run `python gen_main.py --help` for all options.

Scale presets used by the generator are:

- `0`: scale from 1 to 2
- `1`: scale from 2 to 4
- `3`: scale from 1 to 4
- `4`: scale from 4 to 6

The effective number of outputs per complete source pair is `2 * num_sample * cropnum`, because crops are generated in both mapping directions. Existing complete outputs are skipped; incomplete groups are removed and regenerated.

## Utility scripts

- `visualize_registration.py`: visualize registered pairs.
- `check_gt_homography.py`: validate homography records.
- `filter_dataset.py`: remove image names listed in a text file and update the aggregate homography file.
- `copy_png_files.py`: collect the first two PNG files from each child directory and rename them as pairs.

Use `--help` on command-line utilities that accept arguments.

## Notes

- Paths are supplied through command-line arguments; no machine-specific absolute paths are required.
- Input imagery can be large. Generation may require substantial memory and processing time.
- Verify that you have permission to redistribute any generated data and the source datasets.
- Several auxiliary scripts reflect dataset-specific workflows and may require adaptation to your directory structure.


