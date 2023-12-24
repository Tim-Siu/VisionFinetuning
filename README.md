# Fine-tuning Mask R-CNN

## Introduction
This is an project to fine-tune Mask R-CNN on Penn-Fudan Database for Pedestrian Detection and Segmentation. It utilises latest APIs from PyTorch and Torchvision. It references the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) on Torchvision Object Detection Finetuning Tutorial. Adaptation is made to fit remote training on SoC compute clusters.

## Installation and Setup
### Installation without using the sbatch script
1. Clone this repository
2. Install dependencies
```
conda install pytorch torchvision matplotlib pycocotools
```
3. Download the Penn-Fudan dataset
```
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip PennFudanPed.zip
```
4. Run the training script
```
python train.py
```
### Installation using the sbatch script
1. Clone this repository
2. Submit the sbatch script
```
sbatch experiment.sh
```

## Results
The model is trained for 2 epochs. Checkpoints are saved at the end of two epochs. It is released on [HuggingFace](https://huggingface.co/Tim-Xu/MaskRCNN-finetune/blob/main/model.pth). A visualization of the results on a test image is shown below.

![visualization.png](results%2Fvisualization.png)