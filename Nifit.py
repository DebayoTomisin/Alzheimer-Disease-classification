import os
from keras.layers import Input, Dense, Flatten, Dropout, merge, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop, SGD, Adam
from keras import regularizers
from keras import backend as k

import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import nibabel as nib
from sklearn.model_selection import train_test_split
import math
import glob
from matplotlib import pyplot as plt

ff = glob.glob('./niff images/*')

# print(ff[0])

print('the number df files in the data is', len(ff))

# def load_data(path):


images = []
for file in range(len(ff)):
    a = nib.load(ff[file])
    a = a.get_data()
    a = a[:, :, 78:129]
    if a.shape[0] == 256:
        # print(a.shape)
        for i in range(a.shape[2]):
            images.append((a[:, :, i]))

images = np.asarray(images)
print(images.shape)
# print(a[:, :, 0].shape)
# images_arr = np.concatenate(images, axis=0)
# print(images_arr.shape)

images = images.reshape(-1, 256, 256, 1)
print(images.shape)

# normalizing the images
m = np.max(images)
mi = np.min(images)

print(m, mi)

images = (images - mi) / (m - mi)

print(np.min(images))
print()
print(np.max(images))

# splitting the data into train and test

train_x, valid_x, train_ground, valid_ground = train_test_split(images, images, test_size=0.2, random_state=42)

print('Dataset (image) shape: {shape}'.format(shape=images.shape))

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(train_x[0], (256, 256))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_x[0], (256, 256))
plt.imshow(curr_img, cmap='gray')

plt.show()

# building the auto encoder
batch_size = 128
epochs = 100
inChannel = 1
x, y = 256, 256
input_img = Input(shape=(x, y, inChannel))


def autoencoder(imput_img):

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(imput_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)

    #decoder
    conv4 = Conv2D(64, (3, 3), activation='relu',padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2, 2))(conv4)

    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2, 2))(conv5)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
    return decoded


auto_encoder = Model(input_img, autoencoder(input_img))
auto_encoder.compile(loss='mean_squared_error', optimizer=RMSprop())

print(auto_encoder.summary())

auto_encoder_train = auto_encoder.fit(train_x, train_ground, batch_size=batch_size, epochs=epochs, verbose=0,
                                      validation_data=(valid_x, valid_ground))

loss = auto_encoder_train.history['loss']
val_loss = auto_encoder_train.history['val_loss']

epochs = range(100)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



