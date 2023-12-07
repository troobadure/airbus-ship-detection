import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

# ALPHA = 0.8
# GAMMA = 2

# ## intersection over union
# def IoU(y_true, y_pred, eps=1e-6):
#     intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#     union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
#     return -K.mean( (intersection + eps) / (union + eps), axis=0)

# def DiceBCELoss(targets, inputs, smooth=1e-6):    
       
#     #flatten label and prediction tensors
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)
    
#     inputs = K.expand_dims(inputs)
#     targets = K.expand_dims(targets)
    
#     BCE =  binary_crossentropy(targets, inputs)
#     intersection = K.dot(K.transpose(targets), inputs)   
#     dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
#     Dice_BCE = BCE + dice_loss
    
#     return Dice_BCE

# def dice_p_bce(in_gt, in_pred):
#     return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

# def DiceLoss(targets, inputs, smooth=1e-6):
    
#     #flatten label and prediction tensors
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)

#     inputs = K.expand_dims(inputs)
#     targets = K.expand_dims(targets)

#     intersection = K.dot(K.transpose(targets), inputs)
#     dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
#     return 1 - dice

# #metrics
# def dice_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#     union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
#     return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

# def focal_loss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)

#     BCE = K.binary_crossentropy(targets, inputs)
#     BCE_EXP = K.exp(-BCE)
#     focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

#     return focal_loss


alpha = 0.25
gamma = 2

# metrics
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) \
        / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def jacard_similarity(y_true, y_pred):
    """
        Intersection-Over-Union (IoU), also known as the Jaccard Index
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
    return intersection / union

# losses
def jacard_loss(y_true, y_pred):
    """
        Intersection-Over-Union (IoU), also known as the Jaccard loss
    """
    return 1 - jacard_similarity(y_true, y_pred)

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + \
            dice_loss(y_true, y_pred)
    return loss / 2.0

def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) \
        * (weight_a + weight_b) + logits * weight_b

def focal_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(
        logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred
    )

    return tf.reduce_mean(loss)

# combination of focal loss and dice loss
def focal_dice_loss(y_true, y_pred):
    loss = focal_loss(y_true, y_pred) * 10 + dice_loss(y_true, y_pred)
    return loss / 2.0