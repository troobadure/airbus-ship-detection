from keras import models, layers
from config import *

def double_conv(filters, dropout, input):
    c = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(input)
    c = layers.Dropout(dropout)(c)
    c = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c)
    return c

def build_unet_model(filters=[16,32,64,128], dropouts=[0.1,0.1,0.2,0.2]):
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    normalized = layers.Lambda(lambda x: x / 255.0)(inputs)

    # downsample
    skip_connections = []
    for n_filters, dropout in zip(filters, dropouts):
        c = double_conv(n_filters, dropout, normalized)
        skip_connections.append(c)
        normalized = layers.MaxPooling2D((2,2))(c)

    # bottleneck
    b = double_conv(filters[-1]*2, 0, normalized)
    skip_connections = skip_connections[::-1]

    # upsample
    for idx, (n_filters, dropout) in enumerate(zip(filters[::-1], dropouts[::-1])):
        e = layers.Conv2DTranspose(filters=n_filters, kernel_size=(2,2), strides=(2,2), padding='same')(b)
        e = layers.concatenate([skip_connections[idx], e])
        b = double_conv(n_filters, dropout, e)

    # final convolution
    outputs = layers.Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(b)

    return models.Model(inputs=[inputs], outputs=[outputs])

# show summary
if __name__ == '__main__':
    model = build_unet_model()
    from utils.losses import log_cosh_dice_loss, dice_coef
    model.compile(optimizer='adam', loss=log_cosh_dice_loss, metrics=[dice_coef])
    model.summary()
