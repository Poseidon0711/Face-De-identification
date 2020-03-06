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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import os
from keras.backend import tf as ktf
from keras.utils import Sequence
import cv2
from utils import *
from losses import *
from data_loader import *
# from models import *
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import TensorBoard
# import tensorflow as tf
from keras.backend import set_session
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"



def UNETGenerator(input_img_dim, num_output_channels, path=None):
	input_layer = Input(shape=input_img_dim, name="unet_input")
	
	en_1 = Conv2D(kernel_size=(4, 4), filters=64, strides=(2, 2), padding="same")(input_layer)
	en_1 = LeakyReLU(alpha=0.2)(en_1)
	
	en_2 = Conv2D(kernel_size=(4, 4), filters=128, strides=(2, 2), padding="same")(en_1)
	en_2 = BatchNormalization(name='gen_en_bn_2', axis=-1)(en_2)
	en_2 = LeakyReLU(alpha=0.2)(en_2)

	en_3 = Conv2D(kernel_size=(4, 4), filters=256, strides=(2, 2), padding="same")(en_2)
	en_3 = BatchNormalization(name='gen_en_bn_3', axis=-1)(en_3)
	en_3 = LeakyReLU(alpha=0.2)(en_3)

	en_4 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_3)
	en_4 = BatchNormalization(name='gen_en_bn_4', axis=-1)(en_4)
	en_4 = LeakyReLU(alpha=0.2)(en_4)

	en_5 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_4)
	en_5 = BatchNormalization(name='gen_en_bn_5', axis=-1)(en_5)
	en_5 = LeakyReLU(alpha=0.2)(en_5)

	en_6 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_5)
	en_6 = BatchNormalization(name='gen_en_bn_6', axis=-1)(en_6)
	en_6 = LeakyReLU(alpha=0.2)(en_6)
	
	en_7 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_6)
	en_7 = BatchNormalization(name='gen_en_bn_7', axis=-1)(en_7)
	en_7 = LeakyReLU(alpha=0.2)(en_7)

	en_8 = Conv2D(kernel_size=(4, 4), filters=512, strides=(2, 2), padding="same")(en_7)
	en_8 = BatchNormalization(name='gen_en_bn_8', axis=-1)(en_8)
	en_8 = LeakyReLU(alpha=0.2)(en_8)

	# DECODER
	de_1 = UpSampling2D(size=(2, 2))(en_8)
	de_1 = Conv2D(kernel_size=(4, 4), filters=512, padding="same")(de_1)
	de_1 = BatchNormalization(name='gen_de_bn_1', axis=-1)(de_1)
	de_1 = Dropout(0.5)(de_1)
	de_1 = Concatenate(axis=-1)([de_1, en_7])
	de_1 = Activation('relu')(de_1)

	de_2 = UpSampling2D(size=(2, 2))(de_1)
	de_2 = Conv2D(kernel_size=(4, 4), filters=1024, padding="same")(de_2)
	de_2 = BatchNormalization(name='gen_de_bn_2', axis=-1)(de_2)
	de_2 = Dropout(0.5)(de_2)
	de_2 = Concatenate(axis=-1)([de_2, en_6])
	de_2 = Activation('relu')(de_2)

	de_3 = UpSampling2D(size=(2, 2))(de_2)
	de_3 = Conv2D(kernel_size=(4, 4), filters=1024, padding="same")(de_3)
	de_3 = BatchNormalization(name='gen_de_bn_3', axis=-1)(de_3)
	de_3 = Dropout(0.5)(de_3)
	de_3 = Concatenate(axis=-1)([de_3, en_5])
	de_3 = Activation('relu')(de_3)

	de_4 = UpSampling2D(size=(2, 2))(de_3)
	de_4 = Conv2D(kernel_size=(4, 4), filters=1024, padding="same")(de_4)
	de_4 = BatchNormalization(name='gen_de_bn_4', axis=-1)(de_4)
	de_4 = Dropout(0.5)(de_4)
	de_4 = Concatenate(axis=-1)([de_4, en_4])
	de_4 = Activation('relu')(de_4)

	de_5 = UpSampling2D(size=(2, 2))(de_4)
	de_5 = Conv2D(kernel_size=(4, 4), filters=1024, padding="same")(de_5)
	de_5 = BatchNormalization(name='gen_de_bn_5', axis=-1)(de_5)
	de_5 = Dropout(0.5)(de_5)
	de_5 = Concatenate(axis=-1)([de_5, en_3])
	de_5 = Activation('relu')(de_5)

	de_6 = UpSampling2D(size=(2, 2))(de_5)
	de_6 = Conv2D(kernel_size=(4, 4), filters=512, padding="same")(de_6)
	de_6 = BatchNormalization(name='gen_de_bn_6', axis=-1)(de_6)
	de_6 = Dropout(0.5)(de_6)
	de_6 = Concatenate(axis=-1)([de_6, en_2])
	de_6 = Activation('relu')(de_6)

	de_7 = UpSampling2D(size=(2, 2))(de_6)
	de_7 = Conv2D(kernel_size=(4, 4), filters=256, padding="same")(de_7)
	de_7 = BatchNormalization(name='gen_de_bn_7', axis=-1)(de_7)
	de_7 = Dropout(0.5)(de_7)
	de_7 = Concatenate(axis=-1)([de_7, en_1])
	de_7 = Activation('relu')(de_7)

	de_8 = UpSampling2D(size=(2, 2))(de_7)
	de_8 = Conv2D(kernel_size=(4, 4), filters=num_output_channels, padding="same")(de_8)
	de_8 = Activation('sigmoid')(de_8)
	
	unet_generator = Model(inputs=[input_layer], outputs=[de_8], name='unet_generator')
	if path:
		unet_generator.load_weights(path)
	# unet_generator.summary()
	return unet_generator



input_img_dim = (256, 256, 3)
output_channels = 3
output_img_dim = (256, 256, 3)
sub_patch_dim = (64, 64)
vdim = (224, 224)
sdim = (64, 64)
DATA_DIR = '/DATA/alakh/ems_peeps'
nb_patch_patches, patch_gan_dim = num_patches(output_img_dim=output_img_dim, sub_patch_dim=sub_patch_dim)


model_base = UNETGenerator((256,256,3),3)
model_ours = UNETGenerator((256,256,3),3,'/DATA/alakh/ours/gen_weights.h5')
model_ours_sim = UNETGenerator((256,256,3),3,'/DATA/alakh/ours_ssim/gen_weights.h5')
model_ppgan = UNETGenerator((256,256,3),3,'/DATA/alakh/ppgan/gen_weights_epoch.h5')

# datagen = DataGeneratorTest()

for imgp in os.listdir(DATA_DIR):
	ipath = os.path.join(DATA_DIR,imgp)
	img = Image.open(ipath).resize((256,256))
	img = img.convert("RGB")
	# print (np.array(img).shape)
	img = np.array([np.array(img)], dtype=np.float32)/255.0
	# print (img.shape)

	# img = np.array(np.array(Image.open(ipath).resize((256,256))), dtype=np.float32)/255.0

	# Base
	# for img, ipath in datagen:
	# 	imgo = np.array(model_base.predict(img)*255.0, dtype = np.uint8)[0]
	# 	plt.imsave(os.path.join(os.path.join(DATA_DIR, 'base'), ipath.split('/')[-1]), imgo, cmap='Greys_r')
	imgo = np.array(model_base.predict(img)*255.0, dtype = np.uint8)[0]
	plt.imsave(os.path.join(DATA_DIR, 'base_'+imgp), imgo, cmap='Greys_r')
	# Ours
	# for img, ipath in datagen:
	# 	imgo = np.array(model_ours.predict(img)*255.0, dtype = np.uint8)[0]
	# 	plt.imsave(os.path.join(os.path.join(DATA_DIR, 'ours'), ipath.split('/')[-1]), imgo, cmap='Greys_r')
	imgo = np.array(model_ours.predict(img)*255.0, dtype = np.uint8)[0]
	plt.imsave(os.path.join(DATA_DIR, 'our_'+imgp), imgo, cmap='Greys_r')
	# Ours SIM
	# for img, ipath in datagen:
	# 	imgo = np.array(model_ours_sim.predict(img)*255.0, dtype = np.uint8)[0]
	# 	plt.imsave(os.path.join(os.path.join(DATA_DIR, 'ours_sim'), ipath.split('/')[-1]), imgo, cmap='Greys_r')
	imgo = np.array(model_ours_sim.predict(img)*255.0, dtype = np.uint8)[0]
	plt.imsave(os.path.join(DATA_DIR, 'ours_ssim_'+imgp), imgo, cmap='Greys_r')
	# PPGAN
	# for img, ipath in datagen:
	# 	imgo = np.array(model_ppgan.predict(img)*255.0, dtype = np.uint8)[0]
	# 	plt.imsave(os.path.join(os.path.join(DATA_DIR, 'ppgan'), ipath.split('/')[-1]), imgo, cmap='Greys_r')
	imgo = np.array(model_ppgan.predict(img)*255.0, dtype = np.uint8)[0]
	plt.imsave(os.path.join(DATA_DIR, 'ppgan_'+imgp), imgo, cmap='Greys_r')
