import os
import numpy as np
from skimage.io import imread
from keras.preprocessing.image import ImageDataGenerator

import utils.rle as rle
from config import *

#TODO: check augmentations
aug_args = dict(
    featurewise_center = False, 
    samplewise_center = False,
    rotation_range = 45, 
    width_shift_range = 0.1, 
    height_shift_range = 0.1, 
    shear_range = 0.01,
    zoom_range = [0.9, 1.25],  
    horizontal_flip = True, 
    vertical_flip = True,
    fill_mode = 'reflect',
    data_format = 'channels_last'
)
if AUGMENT_BRIGHTNESS:
    aug_args['brightness_range'] = [0.5, 1.5]

image_aug_gen = ImageDataGenerator(**aug_args)

if AUGMENT_BRIGHTNESS:
    aug_args.pop('brightness_range')

label_aug_gen = ImageDataGenerator(**aug_args)

# TODO: sort out BATCH_SIZE and IMG_SCALING usage
def make_image_gen(input_df, batch_size=BATCH_SIZE):
    image_list = list(input_df.groupby('ImageId'))
    image_batch = []
    mask_batch = []
    while True:
        np.random.shuffle(image_list)
        for img_id, img_mask in image_list:
            image = imread(os.path.join(TRAIN_DIR, img_id))
            mask = rle.combine_masks(img_mask['EncodedPixels'].values)
            if IMG_SCALING is not None:
                image = image[::IMG_SCALING[0], ::IMG_SCALING[1]]
                mask = mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            image_batch += [image]
            mask_batch += [mask]
            if len(image_batch)>=batch_size:
                yield np.stack(image_batch, 0), np.stack(mask_batch, 0).astype(np.float32)
                image_batch, mask_batch=[], []
                
def make_aug_gen(input_generator, seed=None):
    for x_batch, y_batch in input_generator:
        batch_size = x_batch.shape[0]

        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        seed = np.random.choice(range(9999))
        x_aug = image_aug_gen.flow(
            x_batch, 
            batch_size = batch_size, 
            seed = seed, 
            shuffle=True
        )
        y_aug = label_aug_gen.flow(
            y_batch, 
            batch_size = batch_size, 
            seed = seed, 
            shuffle=True)

        yield next(x_aug), next(y_aug)