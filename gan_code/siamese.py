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
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tqdm import tqdm
# import tensorflow as tf
# from keras.backend import set_session


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)


ver_mod_l = vgg_face(weights_path = '/DATA/alakh/verificator/vgg_face_weights.h5', num_ids = 67)
ver_mod_r = vgg_face(weights_path = '/DATA/alakh/vgg_face_weights.h5', num_ids = 67)
# ver_mod.trainable = False
verif_model = verif(ver_mod_l, ver_mod_r)
# verif_model.compile(Adam(lr=1e-4, epsilon=1e-6), loss=contrastive_loss, metrics=[contrastive_loss])

verif_model.load_weights("/DATA/alakh/siamese_weights/siam_25_2_19_1_30.h5")

# plot_losses = PlotLosses()

# datagen = DataGenerator(mode='siamese', batch_size = 2)

img_path = ['/DATA/alakh/cropped_data/Rafd090_38_Caucasian_male_fearful_right.jpg','/DATA/alakh/cropped_data/Rafd090_73_Moroccan_male_surprised_right.jpg', 
'/DATA/alakh/cropped_data/Rafd090_73_Moroccan_male_happy_right.jpg']
positives = [( np.array(preprocess_input(Image.open(x).resize((224,224)))),
	np.array(preprocess_input(Image.open(y).resize((224,224)))) ) for x in img_path for y in img_path 
	if x.split('_')[2]==y.split('_')[2]]
negatives = [( np.array(preprocess_input(Image.open(x).resize((224,224)))),
	np.array(preprocess_input(Image.open(y).resize((224,224)))) ) for x in img_path for y in img_path 
	if x.split('_')[2]!=y.split('_')[2]]
img_pairs = np.array(positives + negatives)
labels_pairs = np.array([0.0 for i in range(len(positives))] + [1.0 for i in range(len(negatives))])

print(labels_pairs)
print(img_pairs.shape)

output = verif_model.predict([[img_pairs[7,0]], [img_pairs[7,1]]])
print(output)
# print(ver_mod.predict(np.array([np.array(preprocess_input(Image.open('../data/Rafd090_38_Caucasian_male_fearful_right.jpg').resize((224,224))))])))
# print(ver_mod.predict(np.array([np.array(preprocess_input(Image.open('../data/Rafd090_73_Moroccan_male_surprised_right.jpg').resize((224,224))))])))
print(K.eval(contrastive_loss(labels_pairs,output)))

exit()

# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50)
# checkpoint_callback = ModelCheckpoint('siamese_weights/siam_25_2_19_23_00.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

try:
	verif_model.fit_generator(generator=datagen, use_multiprocessing=True, epochs=101, callbacks=[plot_losses])
	verif_model.save('/DATA/alakh/siamese_weights/siam_25_2_19_1_30.h5')
except KeyboardInterrupt:
	verif_model.save('/DATA/alakh/siamese_weights/siam_26_2_19_1_30.h5')

