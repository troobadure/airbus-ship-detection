import numpy as np
from skimage.morphology import label


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' # no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' # ignore overfilled mask
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

def masks_as_color(mask_list):
    # take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float32)
    scale = lambda x: (len(mask_list)+x+1) / (len(mask_list)*2) # scale the heatmap image to shift 
    for i,mask in enumerate(mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks

def multi_rle_encode(img, **kwargs):
    # encode ships as separated masks
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]
