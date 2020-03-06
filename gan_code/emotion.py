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

emotion_model_path = "/DATA/alakh/Emotion/emotion_model.hdf5"
model = mini_XCEPTION(emotion_model_path)
model.compile(Adam(lr=1e-4, epsilon=1e-6),loss='mse',metrics=['categorical_accuracy'] )

plot_losses = PlotLosses()

datagen = DataGenerator(mode='similar', batch_size = 16)

# img_paths = ['/home/pratik/rishika_alakh/cropped_data/Rafd090_38_Caucasian_male_angry_frontal.jpg', 
# '/home/pratik/rishika_alakh/cropped_data/Rafd090_38_Caucasian_male_disgusted_frontal.jpg',
# '/home/pratik/rishika_alakh/cropped_data/Rafd090_38_Caucasian_male_fearful_frontal.jpg',
# '/home/pratik/rishika_alakh/cropped_data/Rafd090_38_Caucasian_male_happy_frontal.jpg',
# '/home/pratik/rishika_alakh/cropped_data/Rafd090_38_Caucasian_male_sad_frontal.jpg',
# '/home/pratik/rishika_alakh/cropped_data/Rafd090_38_Caucasian_male_surprised_frontal.jpg',
# '/home/pratik/rishika_alakh/cropped_data/Rafd090_38_Caucasian_male_neutral_frontal.jpg']

# img = np.array([rgb2gray(preprocess_input(Image.open(x).resize((64,64)))) for x in img_path])


model.fit_generator(generator=datagen, use_multiprocessing=False, epochs=11, callbacks=[plot_losses])
model.save('/DATA/alakh/emotion_weights/emos_6_4_19_9_00.h5')

