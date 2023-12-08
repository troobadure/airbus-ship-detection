import os
import numpy as np
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from tqdm import tqdm
from skimage.morphology import binary_opening, disk
from skimage.io import imread

from config import *
from utils import losses
from utils.preprocessing import get_masks_df
from utils.rle import masks_as_color, multi_rle_encode

# load and prepare dataframe with masks and image ids
masks_df, balanced_df = get_masks_df()

val_df = pd.merge(masks_df, balanced_df)

# laod a trained model
model = keras.models.load_model(
    LOAD_MODEL_PATH,
    {'log_cosh_dice_loss': losses.log_cosh_dice_loss, 'dice_coef': losses.dice_coef},
    compile=False
)

# save some examples of model predictions
def raw_prediction(img, path=TEST_DIR):
    img = imread(os.path.join(path, img_name))
    img = np.expand_dims(img, 0) / 255.0
    cur_seg = model.predict(img, verbose = 0)[0]
    return cur_seg, img[0]

def smooth(cur_seg):
    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))

def predict(img, path=TEST_DIR):
    cur_seg, c_img = raw_prediction(img, path=path)
    return smooth(cur_seg), c_img

## get a sample of each group of ship count
samples = val_df.groupby('ships').apply(lambda x: x.sample(1))
fig, m_axs = plt.subplots(samples.shape[0], 4, figsize = (15, samples.shape[0]*4))
[c_ax.axis('off') for c_ax in m_axs.flatten()]

for (ax1, ax2, ax3, ax4), img_name in zip(m_axs, samples.ImageId.values):
    first_seg, first_img = raw_prediction(img_name, TRAIN_DIR)
    ax1.imshow(first_img)
    ax1.set_title('Image: ' + img_name)
    ax2.imshow(first_seg[:, :, 0], cmap=get_cmap('jet'))
    ax2.set_title('Model Prediction')
    reencoded = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]))
    ax3.imshow(reencoded)
    ax3.set_title('Prediction Masks')
    ground_truth = masks_as_color(masks_df.query('ImageId=="{}"'.format(img_name))['EncodedPixels'])
    ax4.imshow(ground_truth)
    ax4.set_title('Ground Truth')
    
fig.savefig('validation.png')

# prepare submission
if GENERATE_SUBMISSION:
    test_paths = os.listdir(TEST_DIR)

    def pred_encode(img, **kwargs):
        cur_seg, _ = predict(img)
        cur_rles = multi_rle_encode(cur_seg, **kwargs)
        return [[img, rle] for rle in cur_rles if rle is not None]

    out_pred_rows = []
    for img_name in tqdm(test_paths):
        out_pred_rows += pred_encode(img_name, min_max_threshold=1.0)

    sub = pd.DataFrame(out_pred_rows)
    sub.columns = ['ImageId', 'EncodedPixels']
    sub.to_csv('submission.csv', index=False)

    ## save some samples from the submission
    samples = val_df.groupby('ships').apply(lambda x: x.sample(1))
    TOP_PREDICTIONS = samples.shape[0]
    fig, m_axs = plt.subplots(TOP_PREDICTIONS, 2, figsize = (9, TOP_PREDICTIONS*5))
    [c_ax.axis('off') for c_ax in m_axs.flatten()]

    for (ax1, ax2), img_name in zip(m_axs, sub.ImageId.unique()[:TOP_PREDICTIONS]):
        img = imread(os.path.join(TEST_DIR, img_name))
        img = np.expand_dims(img, 0) / 255.0
        ax1.imshow(img[0])
        ax1.set_title('Image: ' + img_name)
        ax2.imshow(masks_as_color(sub.query('ImageId=="{}"'.format(img_name))['EncodedPixels']))
        ax2.set_title('Prediction')

    fig.savefig('submission.png')