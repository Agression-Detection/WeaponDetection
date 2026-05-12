# Multimodal Aggression Detection: YOLO Spatial Model

## Overview
This repository contains the spatial weapon detection component of our Dual-Stream Aggression Detection System. We utilize a modified YOLOv11 architecture to identify three specific target classes: Handgun, Rifle, and Knife. To prevent catastrophic forgetting of foundational visual features, the training pipeline implements a 3-phase progressive unfreezing strategy alongside discriminative layer-wise learning rates. 

## Dataset
The model is trained and evaluated on the "Weapons Dataset ( YOLO format )".
* **Dataset Link:** [Kaggle: Weapons Dataset (YOLO format)](https://www.kaggle.com/datasets/mukund23/weapons-dataset-yolo-format)

## Code Organization
The repository is structured to separate AWS deployment configurations from the core PyTorch source code:

* `YOLO_Model.ipynb`: A Jupyter Notebook designed for exploratory data analysis and local testing.
* `run.py`: The deployment script that configures and launches the distributed training job on AWS SageMaker using an `ml.g5.12xlarge` instance.
* `requirements.txt`: Contains all necessary Python dependencies.
* `src/` (Core Source Code):
  * `dataset.py`: Defines the `WeaponsDataset` class and `collate_fn`. Applies advanced image augmentations (Mosaic, ColorJitter, MotionBlur) using the `albumentations` library.
  * `train.py`: Contains the PyTorch Distributed Data Parallel (DDP) training loop, automatic mixed precision (AMP) scaling, and custom AdamW optimizer logic for discriminative fine-tuning.
  * `model_t.py`: The evaluation and testing script. Computes Mean Average Precision (mAP), F1-scores, class-specific recall/precision, and generates a visual Confusion Matrix.

## Requirements and Installation
This environment was built and executed using **Python 3.10** and the **PyTorch 2.2.0** framework.

To set up the environment locally, clone the repository and install the dependencies:

```bash
git clone [https://github.com/your-org/YOLO_Model.git](https://github.com/your-org/YOLO_Model.git)
cd YOLO_Model
pip install -r requirements.txt
