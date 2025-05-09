# OBB Augmentation Pipeline

This augmentation pipeline is specialized for single-class oriented bounding boxes in yolo-format utilizing the albumenations library.

The pipeline can use different subsets at the same time. As common for yolo, the images and labels need to follow this structure:

data_dir/images/subset1/subset1_000001.png
data_dir/labels/subset1/subset1_000001.txt

The labels need to normalized and need to be in the format (class x1 y1 x2 y2 x3 y3 x4 y4).


## Installation
### Prerequisites
- Python3
- requirements.txt


## Usage
Start with **augmentation-pipeline.ipynb**