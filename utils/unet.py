""" U-Net model"""
import tensorflow as tf

LAYERS = tf.keras.layers

def conv_block(input_tensor, num_filters):
    encoder = LAYERS.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = LAYERS.BatchNormalization()(encoder)
    encoder = LAYERS.Activation('relu')(encoder)
    encoder = LAYERS.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = LAYERS.BatchNormalization()(encoder)
    encoder = LAYERS.Activation('relu')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = LAYERS.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = LAYERS.Conv2DTranspose(num_filters, (2, 2),
                                     strides=(2, 2),
                                     padding='same')(input_tensor)
    decoder = LAYERS.concatenate([concat_tensor, decoder], axis=-1)
    decoder = LAYERS.BatchNormalization()(decoder)
    decoder = LAYERS.Activation('relu')(decoder)
    decoder = LAYERS.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = LAYERS.BatchNormalization()(decoder)
    decoder = LAYERS.Activation('relu')(decoder)
    decoder = LAYERS.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = LAYERS.BatchNormalization()(decoder)
    decoder = LAYERS.Activation('relu')(decoder)
    return decoder


def model(img_shape=(480, 640, 3),
          num_of_class=22,
          final_activation='softmax'):
    """Implementation of the U-Net """
    inputs = LAYERS.Input(shape=img_shape)
    # 256
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 128
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 64
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 32
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 16
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    # 8
    center = conv_block(encoder4_pool, 1024)
    # center
    decoder4 = decoder_block(center, encoder4, 512)
    # 16
    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32
    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64
    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256
    outputs = LAYERS.Conv2D(num_of_class, (1, 1),
                            activation=final_activation)(decoder0)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='U-Net')
