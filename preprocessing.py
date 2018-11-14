import numpy as np
import scipy.io as scio
import os, re
from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt

import cv2

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_key(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def resize_image(path, new_path):
    filenames = []
    for root, dirname, filenames in os.walk(path):
        filenames.sort(key=natural_key)
        rootpath = root
        #print(filenames)
        print(len(filenames))
        for item in filenames:
            if os.path.isfile(path+item):
                im = Image.open(path+item)
                f, e = os.path.splitext(item)
                imResize = im.resize((128, 128), Image.ANTIALIAS)
                imResize.save(new_path+f+'.jpg', 'JPEG', quality=90)

resize_image("D:\Learnning\TensorFlow\models_example\skin_cancer_detection_segmentation\gt1/", "D:\Learnning\TensorFlow\models_example\skin_cancer_detection_segmentation\gt1_resized/")
resize_image("D:\Learnning\TensorFlow\models_example\skin_cancer_detection_segmentation\melanoma1/", "D:\Learnning\TensorFlow\models_example\skin_cancer_detection_segmentation\melanoma1_resized/")
resize_image("D:\Learnning\TensorFlow\models_example\skin_cancer_detection_segmentation\others1/", "D:\Learnning\TensorFlow\models_example\skin_cancer_detection_segmentation\others1_resized/")

def get_filenames(path):
    filenames = []
    for root, dirname, filenames in os.walk(path):
        filenames.sort(key=natural_key)
        rootpath = root
        print(len(filenames))
        return filenames

root_path = "D:\Learnning\TensorFlow\models_example\skin_cancer_detection_segmentation/"
filenames_melanoma = get_filenames(root_path+"melanoma1_resized/")
filenames_others = get_filenames(root_path+"others1_resized/")
filenames_gt = get_filenames(root_path+"gt1_resized/")
print('filenames_melanoma', filenames_melanoma)
print('filenames_others', filenames_others)

filenames_total = filenames_melanoma + filenames_others
filenames_total.sort(key=natural_key)

total_image = []
for file in filenames_total:
    if os.path.exists(root_path+"melanoma1_resized/"+file):
        total_image.append(ndimage.imread(root_path+"melanoma1_resized/"+file))
    else:
        total_image.append(ndimage.imread(root_path+"others1_resized/"+file))
print('len(total_image)=', len(total_image))

total_image = np.array(total_image)
np.save('total_images.npy', total_image)

gt_images = []
for file in filenames_gt:
    #print('debug1:', (root_path + "gt1_resized/" + file))
    #print('debug2:', os.path.exists(root_path + "gt1_resized/" + file))
    gt_images.append(ndimage.imread(root_path + "gt1_resized/" + file))
    #gt_images = np.array(gt_images)

plt.imshow(gt_images[0], cmap="gray")
#plt.show()

np.unique(gt_images[0])

gt_labels_binary = []
for gt_image in gt_images:
    #gt_images = np.array(gt_images)
    ret, image = cv2.threshold(gt_image, 127, 255, cv2.THRESH_BINARY)
    gt_labels_binary.append(image)

gt_labels_binary = np.array(gt_labels_binary)
np.unique(gt_labels_binary[0])
gt_labels_binary = gt_labels_binary/255
np.unique(gt_labels_binary[0])
np.save('gt_labels_binary.npy', gt_labels_binary)
