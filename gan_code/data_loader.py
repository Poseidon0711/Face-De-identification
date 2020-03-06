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
from random import shuffle



class DataGenerator(Sequence):
	def __init__(self, mode, batch_size):
		self.input_img_dim = (256, 256, 3)
		self.output_channels = 3
		self.output_img_dim = (256, 256, 3)
		self.sub_patch_dim = (64, 64)
		self.vdim = (224, 224)
		self.sdim = (64, 64)
		DATA_DIR = '/DATA/alakh/cropped_data/'
		imgs = os.listdir(DATA_DIR)
		self.imgs = [os.path.join(DATA_DIR, x) for x in imgs if x.split('_')[-2]!='contemptuous']
		shuffle(self.imgs)
		# self.img_pairs = [(x,y) for x in self.imgs for y in self.imgs]
		self.emotion_labels = {'angry':0, 'disgusted':1, 'fearful':2, 'happy':3, 'sad':4, 'surprised':5, 'neutral':6}
		sub_ids = ['01', '02', '03', '04', '05', '07', '08', '09', '10', '11', '12', '14', 
		    '15', '16', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', 
		    '29', '30', '31', '32', '33', '35', '36', '37', '38', '39', '40', '41', '42', 
		    '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', 
		    '56', '57', '58', '59', '60', '61', '63', '64', '65', '67', '68', '69', '70', 
		    '71', '72', '73']

		self.identities = {sub_ids[i]:i for i in range(len(sub_ids))}
		self.mode = mode
		self.batch_size = batch_size

	def __getitem__(self,idx):

		if self.mode=='verificate':
			img_path = self.imgs[idx*self.batch_size:(idx+1)*self.batch_size]
			img = np.array([preprocess_input(Image.open(x).resize((224,224))) for x in img_path])
			id_label = np.zeros((self.batch_size,67), dtype=np.float32)
			for i,ipath in enumerate(img_path):
				id_label[i,self.identities[ipath.split('_')[3]]] = 1.0
			return img, id_label
		elif self.mode=='siamese':
			img_path = self.imgs[idx*self.batch_size:(idx+1)*self.batch_size]
			positives = [( np.array(preprocess_input(Image.open(x).resize((224,224)))),
				np.array(preprocess_input(Image.open(y).resize((224,224)))) ) for x in img_path for y in img_path 
				if x.split('_')[3]==y.split('_')[3]]
			# temp_pos = [x for x in positives]
			negatives = [( np.array(preprocess_input(Image.open(x).resize((224,224)))),
				np.array(preprocess_input(Image.open(y).resize((224,224)))) ) for x in img_path for y in img_path 
				if x.split('_')[3]!=y.split('_')[3]]
			# while (len(positives)<len(negatives)):
			# 	positives = positives + temp_pos
			img_pairs = np.array(positives + negatives)
			labels_pairs = np.array([0.0 for i in range(len(positives))] + [1.0 for i in range(len(negatives))])
			# print (labels_pairs)
			return [img_pairs[:,0], img_pairs[:,1]], labels_pairs
		elif self.mode=='similar':
			img_path = self.imgs[idx*self.batch_size:(idx+1)*self.batch_size]
			img = np.array([rgb2gray(preprocess_input(Image.open(x).resize((64,64)))) for x in img_path])
			emo_label = np.zeros((self.batch_size,7), dtype=np.float32)
			for i,ipath in enumerate(img_path):
				# print(ipath)
				emo_label[i,self.emotion_labels[ipath.split('_')[5]]] = 1.0
			return img, emo_label



	def __len__(self):
		return int(len(self.imgs)*1.0/self.batch_size)


class DataGeneratorTest(Sequence):
	def __init__(self):
		DATA_DIR = '/home/alakh/gan_code/test'
		imgs = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
		self.imgs = [os.path.join(DATA_DIR, x) for x in imgs]
		shuffle(self.imgs)

	def __getitem__(self,idx):
		img_path = [self.imgs[idx]]
		img = np.array([np.array(Image.open(x).resize((256,256))) for x in img_path], dtype=np.float32)/255.0
		return img, img_path[0]

	def __len__(self):
		return len(self.imgs)
