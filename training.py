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
from config import *

WEIGHTS_PATH="seg_model_weights.best.hdf5"
SAMPLES_PER_GROUP = 4000

def main():
    # read csv file with masks encodings
    masks_df = pd.read_csv(MASKS_PATH)
    # mark images with ships
    masks_df['ships'] = masks_df['EncodedPixels'].map(lambda row: 1 if isinstance(row, str) else 0)
    # get number of ships (masks) per image
    unique_img_ids = masks_df.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    masks_df.drop(['ships'], axis=1, inplace=True)

    # some files are too small/corrupt
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(
        lambda img_id: os.stat(os.path.join(TRAIN_DIR, img_id)).st_size/1024
    )
    # keep only files bigger than 50kb
    unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]

    # TODO: consider dropping them all and adding some after first training stage/balance ship pixels count per image
    # instead of balancing number of ships per image, I just drop the majority of images with no ships
    if not BALANCE_SHIP_COUNT:
        balanced_df = unique_img_ids.drop(unique_img_ids[unique_img_ids['ships'] == 0].sample(EMPTY_DROP_COUNT).index)
    else:
        balanced_df = unique_img_ids.groupby('ships').apply(
            lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x
        )

    # split data into train and validation sets
    train_ids, valid_ids = train_test_split(
        balanced_df, test_size = 0.1, stratify = balanced_df['ships']
    )
    train_df = pd.merge(masks_df, train_ids)
    val_df = pd.merge(masks_df, valid_ids)

    # get image data generator for training
    n_train_images, *_ = train_df.shape
    steps_per_epoch = min(MAX_TRAIN_STEPS, n_train_images//BATCH_SIZE)
    aug_gen = make_aug_gen(make_image_gen(train_df))

    # get image data for validation
    val_x, val_y = next(make_image_gen(val_df, VALID_IMG_COUNT))

    #TODO: changle loss, probably metric, and comment
    #TODO: consider changing build_unet_model parameters
    # create unet model with Adam optimizer and focal loss
    unet_model = build_unet_model()
    unet_model.compile(
        optimizer=Adam(1e-3, decay=1e-6), loss=losses.focal_dice_loss,
        metrics=[losses.dice_coef, MeanIoU(num_classes=2, name='mean_iou')]
    )

    # create callbacks for model training
    checkpointer = ModelCheckpoint(
        WEIGHTS_PATH, monitor='val_dice_coef', verbose=1, save_best_only=True, 
        mode='min', save_weights_only=True
    )
    lr_reducer = ReduceLROnPlateau(
        monitor='val_dice_coef', factor=0.33, patience=1, verbose=1, mode='min',
        min_delta=0.0001, cooldown=0, min_lr=1e-8
    )
    early_stopper = EarlyStopping(
        monitor="val_dice_coef", mode="min", verbose=2, patience=30
    )

    # train the model
    loss_history = unet_model.fit(
        aug_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=MAX_TRAIN_EPOCHS,
        validation_data=(val_x, val_y),
        callbacks=[checkpointer, early_stopper, lr_reducer],
        workers=1
    )

    # TODO: change paths with constants
    unet_model.load_weights(WEIGHTS_PATH)
    unet_model.save('seg_model.h5')

    if IMG_SCALING is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
        fullres_model.add(unet_model)
        fullres_model.add(layers.UpSampling2D(IMG_SCALING))
    else:
        fullres_model = unet_model
    fullres_model.save('fullres_model.h5')
    


if __name__ == '__main__':
    main()