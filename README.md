# Airbus Ship Detection using Tensorflow, UNet and Dice

This repository is a solution to Airbus Ship Detection Challenge https://www.kaggle.com/c/airbus-ship-detection/overview implemented as a test task for Winter`24 Data Science Internship at WINSTARS.AI. It features U-Net deep neural network architecture modified with Dropout layers and variations of dice score used for loss and metrics.

## INTRODUCTION

Airbus Ship Detection Challenge is a problem of semantic analysis: every pixel on the image has to be classified as ship or not ship.The dataset consists of ~190k pictures of water surface taken by satellites and a csv file that contains encoded masks for each image indicating ship positions. Images are rgb in a resolution of 768x768. Mask are run-length encoded and one row in the file stands for one separate ship on the image. There can be 0 or multiple ships on the image but masks never overlap.

## SUMMARY

The main challenge of this problem is dataset imbalance. This solution features a basic way to handle it, by dropping imdages without ships and stratifying dataset by number of ships. This is a very basic way to handle it and can certainly be improved by pixel count balancing and smart cropping etc. Nevertheless, even for this implementation U-Net architecture with Dice based loss reached a performace of `val_dice_coef: 0.4184`, `val_binary_accuracy: 0.9935`, `val_log_cosh_dice_loss: 0.1628` and `private score: 0.75324`, `public score: 0.56209`. Which is a decent baseline performace for further development and hypertuning.

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

Valuable thoughts and directions for improvement
https://www.kaggle.com/code/iafoss/unet34-dice-0-87

Basic understanding and some visualizations
https://www.kaggle.com/code/vladivashchuk/notebook6087f5277f/edit

Decent baseline model and visuals
https://www.kaggle.com/code/hmendonca/u-net-model-with-submission/notebook

Losses
https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/tree/master
https://github.com/JunMa11/SegLossOdyssey