import numpy as np
import scipy.io as scio
import os, re
from scipy import ndimage
from matplotlib import pyplot as plt
#%matplotlib inline

import keras

from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras import backend as keras

images = np.load('total_images.npy')
gt_labels = np.load('gt_labels_binary.npy')
#print('image:', images)
#print('gt_label_binary:', gt_labels)
#print('len_image:', len(images))
#print('len_gt_label_binary:', len(gt_labels))

train_indices = np.random.choice(2000, 1900, replace=False)
#print('len_train_indices:', len(train_indices))

train_images = [images[i] for i in train_indices]
train_labels = [gt_labels[i] for i in train_indices]

#print('len_train_image:', len(train_images))
#print('len_train_label:', len(train_labels))

test_indices = [i for i in range(2000) if i not in train_indices]
#print('len_test_indices:', len(test_indices))

test_images = [images[i] for i in test_indices]
test_labels = [gt_labels[i] for i in test_indices]

#print('len_test_image:', len(test_images))
#print('len_test_label:', len(test_labels))

plt.imshow(images[0])
#plt.show()

plt.imshow(gt_labels[0])
#plt.show()

#print(gt_labels.shape)

image_dims = images[0].shape
#print(image_dims)

train_mean = np.mean(train_images, axis=(0, 1, 2, 3))
train_std = np.std(train_images, axis=(0, 1, 2, 3))
train_images = (train_images - train_mean)/(train_std+1e-7)

test_mean = np.mean(test_images, axis=(0, 1, 2, 3))
test_std = np.std(test_images, axis=(0, 1, 2, 3))
test_images = (test_images-test_mean)/(test_std+1e-7)

train_labels = np.expand_dims(train_labels, axis=3)
test_labels = np.expand_dims(test_labels, axis=3)

#print('train_labels.shape:', train_labels.shape)
#print('test_labels.shape:', test_labels.shape)

np.save('train_images.npy', train_images)
np.save('test_images.npy', test_images)
np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_labels)

train_images = np.load('train_images.npy')
test_images = np.load('test_images.npy')
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

#print(train_images[0].shape)
#plt.imshow(train_images[0])
#plt.show()

# plt.imshow(test_images[90])
# plt.show()

#print(train_labels[0].shape)
#plt.imshow(test_labels[1].reshape((128, 192)), cmap="gray")
#plt.show()

##################################################################
#Creat models
def get_unet_model(image_dims):
    inputs = Input((image_dims[0], image_dims[1], image_dims[2]))
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model
#############################################################################
# # Training
model = get_unet_model((128, 128, 3))
# model.summary()
#
# model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
#
# lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
#
# csv_logger = CSVLogger('Unet_lr_e4_bs_10.csv')
#
# model_checkpoint = ModelCheckpoint("Unet_lr_e4_bs_10.hdf5", monitor='val_loss', verbose=1, save_best_only=True)
#
# model.fit(train_images, train_labels, batch_size=4, epochs=25, verbose=1, validation_data=(test_images, test_labels), shuffle=True, callbacks=[lr_reducer, csv_logger, model_checkpoint])
# # finish training
############################################################################
model.load_weights('Unet_lr_e4_bs_10.hdf5')
print(train_images.shape)

test_images = np.expand_dims(test_images, axis=1)

#sample = test_images[1]
#sample = sample.reshape(1, 128, 192, 3)

predictions = model.predict(test_images[90].reshape((1, 128, 128, 3)))
print('predictions.shape:', predictions.shape)

predictions = predictions.reshape((128, 128))
results = predictions > 0.5
print('results.shape:', results.shape)

plt.imshow(results, cmap="gray")
plt.show()
