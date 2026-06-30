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
