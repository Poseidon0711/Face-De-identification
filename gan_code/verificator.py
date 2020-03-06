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
from keras.utils import Sequence
import cv2
from utils import *
from losses import *
from data_loader import *
from models import *
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import TensorBoard
from tqdm import tqdm

model = vgg_face(weights_path = '/DATA/alakh/verificator/vgg_face_weights.h5', num_ids = 67)

model.summary()
exit()

model.compile(Adam(lr=1e-4, epsilon=1e-6), loss='categorical_crossentropy', metrics=['accuracy'] )
plot_losses = PlotLosses()

datagen = DataGenerator(mode='verificate', batch_size = 2)


model.fit_generator(generator=datagen, use_multiprocessing=True, epochs=200, callbacks=[plot_losses])
model.save('/DATA/alakh/verificator_weights/verif_9_5_19_4.h5')
