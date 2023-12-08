import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) \
        / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + \
            dice_loss(y_true, y_pred)
    return loss / 2.0

def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

alpha = 0.25
gamma = 2

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
    loss = focal_loss(y_true, y_pred) * 10 - tf.math.log(dice_loss(y_true, y_pred))
    return loss / 2.0