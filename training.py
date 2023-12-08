import os
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import layers, models
from keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split

from model import build_unet_model
from utils.generators import make_image_gen, make_aug_gen
import utils.losses as losses
from utils.preprocessing import get_masks_df
from config import *

def main():
    # load and prepare dataframe with masks and image ids
    masks_df, balanced_df = get_masks_df()

    # split data into train and validation sets
    train_ids, valid_ids = train_test_split(
        balanced_df, test_size = 0.1, stratify = balanced_df['ships']
    )
    train_df = pd.merge(masks_df, train_ids)
    val_df = pd.merge(masks_df, valid_ids)

    # get image data generator for training
    steps_per_epoch = train_df.shape[0] // BATCH_SIZE
    steps_per_epoch = min(steps_per_epoch, MAX_TRAIN_STEPS)
    train_aug_gen = make_aug_gen(make_image_gen(train_df))

    # get image data for validation
    validation_steps = val_df.shape[0] // BATCH_SIZE
    validation_steps = min(validation_steps, MAX_VAL_STEPS)
    val_aug_gen = next(make_image_gen(val_df))

    #TODO: changle loss, probably metric, and comment
    #TODO: consider changing build_unet_model parameters
    # create unet model with Adam optimizer and focal loss
    unet_model = build_unet_model()
    unet_model.compile(
        optimizer=Adam(1e-4), loss=losses.log_cosh_dice_loss,
        metrics=[losses.dice_coef, MeanIoU(num_classes=2, name='mean_iou')]
    )

    # create callbacks for model training
    checkpointer = ModelCheckpoint(
        BEST_MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min'
    )
    lr_reducer = ReduceLROnPlateau(
        monitor='val_mean_iou', factor=0.33, patience=2, verbose=1,
        min_delta=0.0001, cooldown=0, min_lr=1e-8
    )
    early_stopper = EarlyStopping(
        monitor="val_mean_iou", verbose=2, patience=10, min_delta=0.005
    )

    # train the model
    loss_history = unet_model.fit(
        train_aug_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_aug_gen,
        validation_steps=validation_steps,
        epochs=MAX_TRAIN_EPOCHS,
        callbacks=[checkpointer, early_stopper, lr_reducer]
    )

    if IMG_SCALING is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
        fullres_model.add(unet_model)
        fullres_model.add(layers.UpSampling2D(IMG_SCALING))
    else:
        fullres_model = unet_model
    fullres_model.save(FULLRES_MODEL_PATH)
    

if __name__ == '__main__':
    main()