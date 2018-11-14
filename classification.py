import numpy as np
import scipy.io as scio
import os,re
import itertools
import keras
from scipy import ndimage
from scipy import misc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import *
from keras.layers import  Input, Flatten, Dense, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate, Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from keras import backend as keras

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_key(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

total_images = np.load('total_images.npy')
ground_truth_images = np.load('gt_labels_binary.npy')

root_path = "D:/Learnning/TensorFlow/models_example/skin_cancer_detection_segmentation/"

def get_filenames(path):
    filenames = []
    for root, dirnames, filenames in os.walk(path):
        filenames.sort(key=natural_key)
        root_path = root
    print(len(filenames))
    return filenames

filenames_melanoma = get_filenames(root_path + "melanoma1_resized/")
filenames_others = get_filenames(root_path + "others1_resized/")

filenames_total = filenames_melanoma + filenames_others
filenames_total.sort(key=natural_key)

segmented_images = np.copy(total_images)
x, y, z = segmented_images[0].shape
print('x:{} y:{} z:{}'.format(x, y, z))
for i in range(len(total_images)):
    for j in range(x):
        for k in range(y):
            for l in range(z):
                print('i:{} j:{} k:{} l:{}'.format(i, j, k, l))
                segmented_images[i][j][k][l] = total_images[i][j][k][l] if ground_truth_images[i][j][k][l] == 1 else 0
    misc.imsave(root_path+"segmented_images/segmented_"+filenames_total[i], segmented_images[i])

segmented_images[0].shape

total_images.shape

ground_truth_images.shape

np.save('segmented_images.npy', segmented_images)

segmented_images = np.load('segmented_images.npy')

classification_labels = np.zeros((len(total_images)))
i = 0
for file in filenames_total:
    if os.path.exists(root_path+"melanoma_resized/"+file):
        classification_labels[i]=0
    else:
        classification_labels[i]=1
    i+=1
print('classification_labels', classification_labels)

np.save('classification_labels.npy', classification_labels)

train_indices = np.random.choice(2000, 1900, replace=False)
print('train_indices:', len(train_indices))

train_images = [segmented_images[i] for i in train_indices]
train_labels = [classification_labels[i] for i in train_indices]
print('train_images:', len(train_images))
print('train_labels:', len(train_labels))

test_indices = [i for i in range(2000) if i not in train_indices]
print('test_indices:', len(test_indices))

test_images = [segmented_images[i] for i in test_indices]
test_labels = [classification_labels[i] for i in test_indices]
print('test_images', len(test_images))
print('test_labels', len(test_labels))

plt.imshow(segmented_images[0])
plt.show()

print('classification_labels[0] =', classification_labels[0])

image_dims = segmented_images[0].shape
print('image_dims', image_dims)

train_mean = np.mean(train_images, axis=(0, 1, 2, 3))
train_std = np.std(train_images, axis=(0, 1, 2, 3))
train_images = (train_images - train_mean)/(train_std+1e-7)

test_mean = np.mean(test_images, axis=(0, 1, 2, 3))
test_std = np.std(test_images, axis=(0, 1, 2, 3))
test_images = (test_images-test_mean)/(test_std+1e-7)

np.save('classification_train_images.npy', train_images)
np.save('classification_test_images.npy', test_images)
np.save('classification_train_labels.npy', train_labels)
np.save('classification_test_labels.npy', test_labels)

train_images = np.load('classification_train_images.npy')
test_images = np.load('classification_test_images.npy')
train_images = np.load('classification_train_labels.npy')
test_labels = np.load('classification_test_labels.npy')

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

    flatten1 = Flatten()(conv9)
    dense2 = Dense(1, activation='sigmoid')(flatten1)
    model = Model(inputs=inputs, outputs=dense2)
    return model

model = get_unet_model((128, 192, 3))
model.summary()

model.compile(optimizer=Adam(lr=1-25), loss='binary_crossentropy', metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)

csv_logger = CSVLogger('Unet_lr_e4_bs_10_classifier.csv')
model_checkpoint = ModelCheckpoint("Unet_lr_e4_bs_10_classifier.hdf5", monitor='val_loss', verbose=1, save_best_only=True)

model.fit(train_images, train_labels, batch_size=10, epochs=20, verbose=1, validation_data=(test_images, test_labels), shuffle=True, callbacks=[lr_reducer, csv_logger, model_checkpoint])

#-----------------------------------------------------------
plt.imshow(test_images[5])
plt.show()

test_images = np.expand_dims(test_images,axis=1)
sample_predictions = model.predict(test_images[5].reshape((1, 128, 192, 3)))
predicted_class = sample_predictions[0]>0.5
predicted_class = 0 if predicted_class[0]==True else 1

predicted_class
print(type(predicted_class))

class_labels = {0:'melanoma', 1:'others'}
print(class_labels[predicted_class])

test_images = np.squeeze(test_images, axis=1)
test_predictions = model.predict(test_images)

#----------------------------------------------------------------
