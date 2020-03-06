from keras.layers import Activation, Input, Dropout, Flatten, Dense, Reshape, Lambda, Concatenate, SeparableConv2D, ZeroPadding2D, MaxPool2D
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy, mse, mae
from keras.models import Model, Sequential
from keras.models import load_model
from keras.regularizers import l2
from keras.layers import MaxPooling2D
from keras import layers
from keras.layers import GlobalAveragePooling2D
import keras.backend as K
import numpy as np
import tensorflow as tf
import time
from keras.utils import generic_utils as keras_generic_utils
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.backend import tf as ktf
from losses import generate_patch_gan_loss
import numpy.random as rng


"""
Generators = UNETGenerator(input_img_dim, num_output_channels)
Discriminators = PatchGanDiscriminator(output_img_dim, patch_dim, nb_patches)
Verificators = vgg_face(weights_path=None,num_ids),verif(convnet) 
Similars = mini_XCEPTION(path=None, input_shape=(64,64,1), num_classes=7, l2_regularization=0.01)
"""

def UNETGenerator(input_img_dim, num_output_channels):
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
    # unet_generator.summary()
    return unet_generator



def PatchGanDiscriminator(output_img_dim, patch_dim, nb_patches):
    input_layer = Input(shape=patch_dim)

    num_filters_start = 64
    nb_conv = int(np.floor(np.log(output_img_dim[0]) / np.log(2)))
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    disc_out = Conv2D(kernel_size=(4, 4), filters=64, strides=(2, 2), padding="same", name='disc_conv_1')(input_layer)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    for i, filter_size in enumerate(filters_list[1:]):
        name = 'disc_conv_{}'.format(i+2)

        disc_out = Conv2D(kernel_size=(4, 4), filters=filter_size, strides=(2, 2), padding="same", name=name)(disc_out)
        disc_out = BatchNormalization(name=name + '_bn', axis=-1)(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    disc_out = Flatten()(disc_out)
    disc_out = Dense(2, activation = 'softmax', name = 'realistic')(disc_out)

    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=disc_out,
                                                      patch_dim=patch_dim,
                                                      input_layer=input_layer,
                                                      nb_patches=nb_patches)
    return patch_gan_discriminator


def vgg_face(weights_path=None,num_ids=67):
    img = Input(shape=(224, 224, 3))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPool2D((2, 2), strides=(2, 2))(conv1_2)
    
    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPool2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPool2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPool2D((2, 2), strides=(2, 2))(conv5_3)

    fc6 = Conv2D(4096, (7, 7), activation='relu', name='fc6')(pool5)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Conv2D(4096, (1, 1), activation='relu', name='fc7')(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    fc8 = Conv2D(2622, (1, 1), name='fc8')(fc7_drop)
    flat = Flatten()(fc8)
    out = Activation('softmax')(flat)

    model = Model(input=img, output=out)

    if weights_path:
        model.load_weights(weights_path)

    model.layers.pop()
    model.layers.pop()
    model.layers.pop()


    for layer in model.layers:
        layer.trainable = False

    conv= Conv2D(num_ids, (1, 1), name='fc8')(model.layers[-1].output)
    flatLayer=Flatten()(conv)
    out1=Activation('softmax')(flatLayer)
    model1=Model(input=img,output=out1)

    # model1.summary()
    
    return model1



def mini_XCEPTION(path=None, input_shape=(64,64,1), num_classes=7, l2_regularization=0.001):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
               kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(inputs=img_input, outputs=output)

    if(path):
        model.load_weights(path)

    model.layers.pop()
    model.layers.pop()
    model.layers.pop()

    for layer in model.layers:
        layer.trainable = False

    x=Conv2D(num_classes, (3, 3), kernel_regularizer=regularization, padding='same')(model.layers[-1].output)
    x=GlobalAveragePooling2D()(x)
    output=Activation('softmax', name='predictions')(x)

    model = Model(inputs=img_input, outputs=output)

    model.summary()

    return model


def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)


def verif(convnetl, convnetr):
    input_shape = (224, 224, 3)
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    # #build convnet to use in each siamese 'leg'
    # convnet = Sequential()
    # convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
    #                    kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
    # convnet.add(MaxPooling2D())
    # convnet.add(Conv2D(128,(7,7),activation='relu',
    #                    kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
    # convnet.add(MaxPooling2D())
    # convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    # convnet.add(MaxPooling2D())
    # convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    # convnet.add(Flatten())
    # convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
    #encode each of the two inputs into a vector with the convnet
    encoded_l = convnetl(left_input)
    encoded_r = convnetr(right_input)
    # encoded_l.summary()
    #merge two encoded inputs with the l1 distance between them
    # L1_distance = lambda x: K.abs(x[0]-x[1])
    # siamese_net = Sequential()
    dist = Lambda( lambda x: 100*K.mean(K.abs(x[0]-x[1]))) ([encoded_l,encoded_r])
    # prediction = Dense(2,activation='sigmoid',bias_initializer=b_init)(both)
    siamese_net = Model(inputs=[left_input,right_input],outputs=dist)
    # siamese_net.summary()
    return siamese_net


def DCGAN(generator_model, discriminator_model, verif_model, sim_model, input_img_dim, patch_dim, vdim, sdim):
# def DCGAN(generator_model, discriminator_model, verif_model, input_img_dim, patch_dim, vdim):
    generator_input = Input(shape=input_img_dim, name="DCGAN_input")
    
    generated_image = generator_model(generator_input)
    # output_image = Lambda(lambda image: image*255.0)(generated_image)
    # verif_input = Lambda(lambda image: [ktf.image.resize_images(image[0], vdim), ktf.image.resize_images(image[1], vdim)], 
    #     name = 'vi')([generator_input, generated_image])
    verif_input = Lambda(lambda image: ktf.image.resize_images(image, vdim), name = 'vi')(generated_image)
    sim_input = Lambda(lambda image: ktf.image.resize_images(image, sdim), name = 'tsi')(generated_image)
    sim_input = Lambda(lambda image: ktf.image.rgb_to_grayscale(image), name = 'si')(sim_input)
    
    h, w = input_img_dim[:-1]
    ph, pw = patch_dim

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(int(h / ph))]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(int(w / pw))]
    
    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1],
                col_idx[0]:col_idx[1], :], output_shape=input_img_dim)(generated_image)
            list_gen_patch.append(x_patch)

    dcgan_output = discriminator_model(list_gen_patch)
    verif_output = verif_model(verif_input)
    sim_output = sim_model(sim_input)
    
    dc_gan = Model(inputs=[generator_input], outputs=[dcgan_output, verif_output, sim_output, generated_image, generated_image], name="DCGAN")
    # dc_gan = Model(inputs=[generator_input], outputs=[dcgan_output, verif_output, generated_image, generated_image], name="DCGAN")
    return dc_gan
