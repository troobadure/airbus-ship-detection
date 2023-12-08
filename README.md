# Airbus Ship Detection using Tensorflow, U-Net and Dice

This repository is a solution for [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview) implemented as a test task for Winter`24 Data Science Internship at WINSTARS.AI.  
It features U-Net deep neural network architecture modified with Dropout layers and variations of dice score used for loss and metrics.

## DESCRIPTION

Airbus Ship Detection Challenge is a problem of semantic segmentation: every pixel of the image has to be classified as ship or not ship. The dataset consists of ~190k pictures of water surface taken by satellites and a csv file that contains encoded masks for each image indicating ship positions. Images are rgb in a resolution of 768x768. Mask are run-length encoded and one row in the file stands for one separate ship on the image. There can be 0 or multiple ships on the image but masks never overlap. The majority of pixels are background.

## SUMMARY

The main challenge of this problem is dataset imbalance. This solution features a basic way to handle it, by dropping images without ships and stratifying dataset by number of ships. It is effective but can certainly be improved by pixel count balancing, smart cropping etc.  
Nevertheless, even for this implementation U-Net architecture with Dice based loss reached a performace of `val_dice_coef: 0.4184`, `val_binary_accuracy: 0.9935`, `val_log_cosh_dice_loss: 0.1628` and `private score: 0.75324`, `public score: 0.56209`. Which is a decent baseline performace for further development and hypertuning.

## DEPLOYMENT

1. `git clone https://github.com/troobadure/airbus-ship-detection.git`
2. `cd airbus-ship-detection`
3. `python -m venv env; env/Scripts/activate`
4. `pip install -r requirements.txt`
5. `mkdir input`
5. Extract [airbus-ship-detection.zip](https://www.kaggle.com/competitions/airbus-ship-detection/data) contents into `input` folder.
6. `python inference.py`

## CONTENTS

- [`eda.ipynb`](eda.ipynb) - Exploratory data analysis on dataset, some preprocessing. Independent from training and inference.
- [`model.py`](model.py) - U-Net model implementation. Run to see model summary.
- [`training.py`](training.py) - Preprocessing and model training.
- [`inference.py`](inference.py) - Model inference. Run to see samples of predictions.
- [`config.py`](config.py) - Configuration parameters for model building, training and inference.
- [`utils`](utils) - Folder with utility scripts used in training and inference.
- [`outputs`](outputs) - Folder containing trained model and inference outputs. 

## PREPROCESSING

Briefly, preprocessing includes such steps:
- loading masks data from csv
- dropping corrupt images
- dropping images with no ships
- balancing dataset by ship count with a maximum of 2000 images per count
- combining mask of separate ships into one example
- splitting on training and validation sets in proportion of 9:1
- building image generator with scaling 768x768 to 256x256
- wrapping it in an augmentation generator that would shear and flip the images 

To address the problem of class imbalance, simple approach is used: removing all the empty images and stratification by ship count. This dataset manipulation gives a huge boost to model training but it is not perfect, since even multiple ships can often occupy only few pixels on the image. So it would be much better to try and balance images by ship pixel count instead of ship count. Perfect way to do it would be cropping images and sampling the results on pixel count basis. It also would be a good idea to take the model after main training part and use previously dropped empty images to finetune it as on difficult examples.

## MODEL

Since the goal is semantic segmentation, U-Net, wich was originally introduced as medical imaging segmentation technique, can be a perfect choice. The original model consists encoder for downsampling, decoder for upsampling and skip connections between every pair of nodes.

![U-Net Architecture](https://b2633864.smushcdn.com/2633864/wp-content/uploads/2022/02/1_unet_architecture_paper.png?size=630x350&lossy=2&strip=1&webp=1)

The model in this solution is a conventional U-Net architecture with slight adjustments:
- Dropouts added at every encoder and decoder step between two convolutions with ratios `[0.1,0.1,0.2,0.2]`
- Conv2DTranspose layer is chosen as upsampling method
- A normalization layer is added at the beginning
- Model inputs and outputs are in 256x256 resolution
- Numbers of filters are set to `[16,32,64,128]`
Details in [`model.py`](model.py)

Possible improvement could be gained by replacing Dropouts with BatchNormalization, using more effective normalization methods at model entrance and adjusting layer numbers. Also, it might boost the performance greatly, to use a second classification model to detect images with no ships and toss them out. Test time augmentation is an alternative way of solving it.

## LOSS

Because of the dataset imbalance that is quite difficult to cope with, loss function is one of the most crucial parts of the solution. Simple loss functions such as Binary Cross-Entropy loss do not work here. Something more specific to image segmentation is needed:
- Dice loss - measures the overlap between the predicted and target segmentation masks, is differentiable, particularly effective when dealing with imbalanced datasets and when the focus is on capturing fine details in the segmentation masks
- IoU (Jaccard Loss) - similar to Dice loss, used in tasks where accurate boundary delineation is critical
- Focal Loss - addresses the problem of class imbalance and focuses on challenging or misclassified samples

In this solution Log Cosh Dice loss is used as it is a smoother variation of usual Dice loss and prooves to be effective. Dice coefficient and IoU are used as supplementary metrics for convenience and tracking.

Possible improvement would be to mix in Focal Loss as it directly addresses main problem of this challenge - class imbalance. 

## POSSIBLE IMPROVEMENTS

- Cropping images in a way to balance ship area throughout the dataset
- Combining focal loss and dice loss for model training
- Ensembling with a Classifier ship/no-ship model
- Adding BatchNormalization, adjusting number of layers
- Test time augmentation
- Post training on previously dropped empty images

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