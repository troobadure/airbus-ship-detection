import os
import pandas as pd
from config import *

def get_masks_df():
    # read csv file with masks encodings
    masks_df = pd.read_csv(MASKS_PATH)
    # mark images with ships
    masks_df['ships'] = masks_df['EncodedPixels'].map(lambda row: 1 if isinstance(row, str) else 0)
    # get number of ships (masks) per image
    unique_img_ids = masks_df.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    masks_df.drop(['ships'], axis=1, inplace=True)

    # some files are too small/corrupt
    if REMOVE_CORRUPT:
        unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(
            lambda img_id: os.stat(os.path.join(TRAIN_DIR, img_id)).st_size/1024
        )
        # keep only files bigger than 50kb
        unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]

    # TODO: change to balancing ship pixels count per image
    # drop the majority of images with no ships and undersample images with low number of ships
    balanced_df = unique_img_ids.drop(unique_img_ids[unique_img_ids['ships'] == 0].sample(EMPTY_DROP_COUNT).index)

    if BALANCE_SHIP_COUNT:
        balanced_df = balanced_df.groupby('ships').apply(
            lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x
        )

    return masks_df, balanced_df