# Airbus Ship Detection using Tensorflow, UNet and Dice

This repository is a solution to Airbus Ship Detection Challenge https://www.kaggle.com/c/airbus-ship-detection/overview implemented as a test task for Winter`24 Data Science Internship at WINSTARS.AI. It features U-Net deep neural network architecture modified with Dropout layers and variations of dice score used for loss and metrics.

## INTRODUCTION

Airbus Ship Detection Challenge is a problem of semantic analysis: every pixel on the image has to be classified as ship or not ship.The dataset consists of ~190k pictures of water surface taken by satellites and a csv file that contains encoded masks for each image indicating ship positions. Images are rgb in a resolution of 768x768. Mask are run-length encoded and one row in the file stands for one separate ship on the image. There can be 0 or multiple ships on the image but masks never overlap.


## SUMMARY


## DEPLOYMENT

1. `git clone https://github.com/troobadure/airbus-ship-detection.git`
2. `python -m venv env; env/Scripts/activate`
3. `pip install -r requirements.txt`
2. `python inference.py`

## CONTENTS
- `eda.ipynb` - Exploratory data analysis on dataset, some preprocessing. Independent from training and inference.
- `model.py` - U-Net model implementation. Run to see model summary.
- `training.py` - Preprocessing and model training.
- `inference.py` - Model inference. Run to see samples of predictions.
- `config.py` - Configuration parameters for model building, training and inference.
- `utils/` - Folder with utility scripts used in training and inference.

## EDA

## PREPROCESSING

## MODEL SELECTION

## TRAINING

## INFERENCE

## IMPROVEMENTS

## REFERENCES