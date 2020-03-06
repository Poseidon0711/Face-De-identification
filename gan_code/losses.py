from keras.layers import Activation, Input, Dropout, Flatten, Dense, Reshape, Lambda, Concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy, mse, mae
from keras.models import Model
from keras.models import load_model
import keras.backend as K
import numpy as np
from PIL import Image
import tensorflow as tf
import time
from keras.utils import generic_utils as keras_generic_utils
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
from keras.backend import tf as ktf


def generate_patch_gan_loss(last_disc_conv_layer, patch_dim, input_layer, nb_patches):
    list_input = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]

    disc = Model(inputs = [input_layer], outputs = [last_disc_conv_layer], name='patch_gan')

    x = [disc(patch) for patch in list_input]
    if len(x) > 1:
        x = Concatenate(name = 'merged')(x)
    else:
        x = x[0]
    x = Dense(2,activation = 'softmax', name='final')(tf.convert_to_tensor(x))

    disc_final = Model(inputs = list_input, outputs = [x], name='discriminator')
    return disc_final


# def contrastive_loss(y_true, y_pred):
#     margin = 1
#     sqaure_pred = K.square(y_pred)
#     margin_square = K.square(K.maximum(margin - y_pred, 0))
#     return K.max(K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square),0)

def contrastive_loss(y_true, y_pred):
    margin = 2
    return K.mean((1-y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))


def ssim_loss(y_true, y_pred):
    return (1-tf.image.ssim(y_true, y_pred, max_val=255.0))/2.0

