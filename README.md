# CASGM
CASGM: Co-Visible Area Guided Optical-SAR Matching with Significant Geometric Differences


# GeoMS-OS Test Dataset

This directory contains paired optical and SAR test data organized by scale and
data subset. Each sample consists of an optical image, a SAR image, and a
homography matrix that maps coordinates from the optical image to the SAR image.

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

Each subset contains a `test` directory with the following subdirectories:

- `opt/`: optical images.
- `sar/`: SAR images.
- `H/`: transformation matrices from optical image coordinates to SAR image
  coordinates.

## File Naming

Files with the same stem form one matched sample:

```text
test/
+-- H/
|   +-- pair_00001.txt
+-- opt/
|   +-- pair_00001.png
+-- sar/
    +-- pair_00001.png
```

For example:

- `opt/pair_00001.png` is the optical image.
- `sar/pair_00001.png` is the corresponding SAR image.
- `H/pair_00001.txt` is the transformation matrix for this pair.
