import numpy as np


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten() # T is needed to align to RLE direction
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T # T is needed to align to RLE direction

def combine_masks(img_masks, shape=(768, 768)):
    '''
    masks: pd dataframe with image_id and EncodedPixels columns
    image_id: the image id to get the masks for
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros(shape, dtype=np.uint8)
    for mask in img_masks:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)

    return np.expand_dims(all_masks, -1)

# def multi_rle_encode(img):
#     labels = label(img[:, :, 0])
#     return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# demonstration
if __name__ == '__main__':
    import pandas as pd
    from skimage.io import imread
    import matplotlib.pyplot as plt
    import os
    print(os.listdir("input"))


    train = os.listdir('input/train')
    print(len(train))

    test = os.listdir('input/test')
    print(len(test))


    masks = pd.read_csv('input/train_ship_segmentations.csv')
    print(masks.head())


    ImageId = '000155de5.jpg'

    img = imread('input/train/' + ImageId)
    all_masks = combine_masks(masks, ImageId)

    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()