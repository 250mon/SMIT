# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:15:27 2020

@author: VCAR_MSI
"""

import tensorflow as tf

#AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Retrieve the images : 
# You can use an archive of creative-commons licensed flower photos from Google.

import pathlib
data_dir = tf.keras.utils.get_file(
                      origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                      fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)


print(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print("image_count=", image_count)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)

roses = list(data_dir.glob('roses/*'))

for image_path in roses[:3]:
    display.display(Image.open(str(image_path)))
    

# Load using keras.preprocessing
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
      
image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)
    
print("label_batch=====>")
print(label_batch)    
