# CASGM
CASGM: Co-Visible Area Guided Optical-SAR Matching with Significant Geometric Differences

&gt; **📌 数据公开说明**
&gt; - **GeoMS-OS Test Dataset（本目录）**：已公开，下载链接见下方 [Dataset Download](#dataset-download)。
&gt; - **CASGM 数据集** 及 **GeoMS-OS 的其余数据（如训练集、完整版等）**：待论文正式发表后公开，敬请期待。

---

# GeoMS-OS Test Dataset

This directory contains paired optical and SAR test data organized by scale and
data subset. Each sample consists of an optical image, a SAR image, and a
homography matrix that maps coordinates from the optical image to the SAR image.

## Dataset Download

- **GeoMS-OS Test Dataset (Public)**  
  下载链接：[https://huggingface.co/datasets/PuYuchi/GeoMS-OS]  
  该链接包含以下 `scale12` 与 `scale24` 的测试子集。

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
