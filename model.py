import tensorflow as tf
import tensorflow.keras.layers as layers

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3


def double_conv(filters, dropout, input):
    c = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(input)
    c = layers.Dropout(dropout)(c)
    c = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_normal')(c)
    return c

def unet_model(in_channels=3, out_channels=1, filters=[64,128,256,512], dropouts=[0.1,0.1,0.2,0.2]):
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    normalized = layers.Lambda(lambda x: x / 255.0)(inputs)

    # down
    skip_connections = []
    for n_filters, dropout in zip(filters, dropouts):
        c = double_conv(n_filters, dropout, normalized)
        skip_connections.append(c)
        normalized = layers.MaxPooling2D((2,2))(c)

    # bottleneck
    b = double_conv(filters[-1]*2, 0, normalized)
    skip_connections = skip_connections[::-1]

    # up
    for idx, (n_filters, dropout) in enumerate(zip(filters[::-1], dropouts[::-1])):
        e = layers.Conv2DTranspose(filters=n_filters, kernel_size=(2,2), strides=(2,2), padding='same')(b)
        e = layers.concatenate([skip_connections[idx], e])
        b = double_conv(n_filters, dropout, e)

    # Final convolution
    outputs = layers.Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(b)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
