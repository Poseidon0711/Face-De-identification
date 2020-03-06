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
from models import *
from keras.optimizers import SGD, Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import TensorBoard
# import tensorflow as tf
from keras.backend import set_session
import os
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]


input_img_dim = (256, 256, 3)
output_channels = 3
output_img_dim = (256, 256, 3)
sub_patch_dim = (64, 64)
vdim = (224, 224)
sdim = (64, 64)

nb_patch_patches, patch_gan_dim = num_patches(output_img_dim=output_img_dim, sub_patch_dim=sub_patch_dim)

gen_nn = UNETGenerator(input_img_dim=input_img_dim, num_output_channels=output_channels)
dis_nn = PatchGanDiscriminator(output_img_dim=output_img_dim, patch_dim=patch_gan_dim, nb_patches=nb_patch_patches)

verif = load_model("/DATA/alakh/verificator_weights/verif_10_2_19_10_00.h5")
verif.name = 'verificator'
verif.trainable = False
for layer in verif.layers:
    layer.trainable = False


sim = load_model("/DATA/alakh/emotion_weights/emos_12_2_19_9_00.h5")
sim.name = 'similar'
sim.trainable = False
for layer in sim.layers:
    layer.trainable = False

dis_nn.trainable = False

opt_discriminator = Adam(lr=1E-5, epsilon=1e-6)
opt_dcgan = Adam(lr=1E-5, epsilon=1e-6)

gen_nn.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

gan_nn = DCGAN(gen_nn, dis_nn, verif, sim, input_img_dim, sub_patch_dim, vdim, sdim)

loss = ['binary_crossentropy', contrastive_loss, categorical_crossentropy, 'mse', ssim_loss]

loss_weights = [1, 1E5, 0.5, 2, 0.25]
gan_nn.compile(loss=loss, loss_weights = loss_weights, optimizer=opt_dcgan)

dis_nn.trainable = True

dis_nn.compile(loss='binary_crossentropy', optimizer=opt_discriminator)



DATA_DIR = '/DATA/alakh/cropped_data/'
imgs = os.listdir(DATA_DIR)
imgs = [x for x in imgs if x.split('_')[4]!='contemptuous']

images = []
verif_labels = []
sim_labels = []
emotion_labels = {'angry':0, 'disgusted':1, 'fearful':2, 'happy':3, 'sad':4, 'surprised':5, 'neutral':6}
sub_ids = ['01', '02', '03', '04', '05', '07', '08', '09', '10', '11', '12', '14', 
    '15', '16', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', 
    '29', '30', '31', '32', '33', '35', '36', '37', '38', '39', '40', '41', '42', 
    '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', 
    '56', '57', '58', '59', '60', '61', '63', '64', '65', '67', '68', '69', '70', 
    '71', '72', '73']

identities = {sub_ids[i]:i for i in range(len(sub_ids))}


for img in imgs:
    imgg = Image.open(DATA_DIR + img).resize((256,256))
    im = np.array(imgg)
    imgg.close()
    im = np.array(im, np.float32)/255.0
    images.append(im)

    emo_label = img.split('_')[4]
    label = np.zeros(7, dtype=np.float32)
    label[emotion_labels[emo_label]] = 1.0
    sim_labels.append(label)

    index=img.split('_')[1]
    label_id = np.zeros(67, dtype=np.float32)
    label_id[identities[index]] = 1.0
    verif_labels.append(label_id)



images = np.array(images)
verif_labels = np.array(verif_labels)
sim_labels = np.array(sim_labels)


nb_epoch = 100
batch_size = 8
n_batches_per_epoch = int((len(images))/batch_size)

D_log_loss =1000
gan_total_loss = 1000
dis_loss = 1000
ver_loss = 1000
sim_loss = 1000
gl_loss = 1000
d = 1000

print ("Training...")


Epoch = []
G_loss_f = []
glossf = 0
D_loss_f = []
dlossf = 0
Ver_loss_f = []
vlossf = 0
Em_loss_f = []
emlossf = 0
Sim_loss_f = []
simlossf = 0


for epoch in range(0, nb_epoch):
	start = time.time()
	print ("Epoch", epoch+1)
	print (n_batches_per_epoch)
	progbar = keras_generic_utils.Progbar(n_batches_per_epoch)
	batch_counter = 1

	for mini_batch_i in range(0, n_batches_per_epoch):
		# print (mini_batch_i)
		sim.trainable = False
		for layer in sim.layers:
		    layer.trainable = False

		verif.trainable = False
		for layer in verif.layers:
		    layer.trainable = False

		X_images = images[mini_batch_i*batch_size:(mini_batch_i+1)*batch_size]
		# try:
		X_discriminator, y_discriminator = get_disc_batch(X_images, gen_nn, sub_patch_dim, batch_counter)

		disc_loss = dis_nn.train_on_batch(X_discriminator, y_discriminator)

		y_gen = np.zeros((X_images.shape[0],2), dtype=np.float32)
		y_gen[:,1] = 1.0
		dis_nn.trainable = False

		gen_loss = gan_nn.train_on_batch(X_images, 
		    [y_gen, verif_labels[mini_batch_i*batch_size:(mini_batch_i+1)*batch_size], 
		    sim_labels[mini_batch_i*batch_size:(mini_batch_i+1)*batch_size], X_images, X_images])

		dis_nn.trainable = True

		D_log_loss = disc_loss
		dlossf+=D_log_loss
		gan_loss = gen_loss[0].tolist()
		dis_loss = gen_loss[1].tolist()
		glossf+=dis_loss
		ver_loss = gen_loss[2].tolist()
		vlossf+=ver_loss
		em_loss = gen_loss[3].tolist()
		emlossf+=em_loss
		cg_loss = gen_loss[4].tolist()
		sim_loss = gen_loss[5].tolist()
		simlossf+=sim_loss


		progbar.add(1, values=[("D", D_log_loss),
		                                ("tot", gan_loss),
		                                ("Gen", dis_loss),
		                                ("v", ver_loss),
		                                ("em", em_loss),
		                                ("cG", cg_loss),
		                                ("s", sim_loss)
		                                ])

		if ((epoch+1)%5 == 0):
			plot_generated_batch(X_images, 'ours_ssim', gen_nn, epoch, 'tng', mini_batch_i)
		batch_counter+=1

	Epoch.append(epoch)
	dlossf=dlossf*1.0/n_batches_per_epoch
	D_loss_f.append(dlossf)
	glossf=glossf*1.0/n_batches_per_epoch
	G_loss_f.append(glossf)
	vlossf=vlossf*1.0/n_batches_per_epoch
	Ver_loss_f.append(vlossf)
	emlossf=emlossf*1.0/n_batches_per_epoch
	Em_loss_f.append(emlossf)
	simlossf=simlossf*1.0/n_batches_per_epoch
	Sim_loss_f.append(simlossf)

	# if ((epoch+1)%5 == 0):
	plt.clf()
	plt.plot(Epoch, D_loss_f, label="Discriminator")
	plt.plot(Epoch, G_loss_f, label="Generator")
	plt.plot(Epoch, Ver_loss_f, label="Identity Similarity")
	plt.plot(Epoch, Em_loss_f, label="Emotion Similarity")
	plt.plot(Epoch, Sim_loss_f, label="Structural Similarity")
	plt.legend()
	plt.savefig('/DATA/alakh/ours_ssim/ourgraph.png')


	gen_weights_path = '/DATA/alakh/ours_ssim/gen_weights.h5'
	gen_nn.save_weights(gen_weights_path, overwrite=True)

	# disc_weights_path = '/DATA/alakh/ours_ssim/disc_weights.h5'
	# dis_nn.save_weights(disc_weights_path, overwrite=True)

	# DCGAN_weights_path = '/DATA/alakh/ours_ssim/DCGAN_weights.h5'
	# gan_nn.save_weights(DCGAN_weights_path, overwrite=True)

