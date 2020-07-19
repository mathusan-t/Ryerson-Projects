import pandas as pd
import numpy as np

# Image preprocessing libraries
from PIL import Image
import skimage
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# GPU access libraries creation libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from numba import cuda

# Model creation libraries
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import *

from keras.utils import to_categorical

# Code used to tell the program to use the GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# the model won't work unless the input has three channels (one for each of RGB). The MRI images are grayscale; we make them RGB by giving the same scalar value for all three channels, that the original image had. 
def grayscale_to_rgb(filename, directory):
    name = ("{}/{}".format(directory, filename))
    X = Image.open(name)
    X = np.asarray(X)
    RGB_X = np.stack((X,)*3, axis = -1)
    return RGB_X

# This functions puts the last two functions together to preprocess our image data. It takes each image as input,
# converts it from grayscale to RGB, and conducts the image entropy calculation on the slices.
def preprocessing_General():
    finalX = []
    finalY = []

    directory = "/home/mathusan/MRP Work/Marcia Data/HC Patients"
    for folder in os.listdir(directory):
        insidefolder = directory + "/" + folder
        for file in os.listdir(insidefolder):
            img = grayscale_to_rgb(file, insidefolder)
            finalX.append(img)
            finalY.append(0)


    directory = "/home/mathusan/MRP Work/Marcia Data/AD Patients"
    for folder in os.listdir(directory):
        insidefolder = directory + "/" + folder
        for file in os.listdir(insidefolder):
            img = grayscale_to_rgb(file, insidefolder)
            finalX.append(img)
            finalY.append(1)

    finalX = np.array(finalX)
    finalY = np.array(finalY)

    return finalX, finalY

# Inception needed images to be at least 299 x 299 in order for the model to accept the image
def resize_Inception(X):
    InceptionX = []
    for i in range(len(X)):    
        img = resize(X[i], (299, 299))
        InceptionX.append(img)
    InceptionX = np.array(InceptionX)
    return InceptionX


# The following lines of code take our input images and outputs X and Y
X, Y = preprocessing_General()
X = resize_Inception(X)
X = X.reshape(len(X), 299, 299, 3)

# This shuffles our data and splits it into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# testing with imagedatagen to see if there is any improvement:
datagen = ImageDataGenerator(
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# This block of code is creating out model. All we had to do is define the input size, and replace the fully connected layers
# since we chose to leave include_top = False. If it was true, it would imply that our problem had the same number of classes as the 
# ImageNet problem, which it does not. ImageNet has 1000 classes; we only have 2. 
X_input = Input(shape = (299, 299, 3))

inceptionBlock = InceptionV3(include_top = False, weights = 'imagenet', input_tensor = X_input, input_shape = (299, 299, 3))
flatten = Flatten()
x = flatten(inceptionBlock.output)
# we took out all of the regular VGG output dense layers, need to incorporate those again
layer1 = Dense(512, activation='relu')(x)
layer2 = Dense(512, activation='relu')(layer1)
layer3 = Dense(1, activation = 'sigmoid')(layer2)

model = Model(X_input, layer3)

sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#test if model saves before even training it
# model.save('before_fit')
    
# train the model and compare each iteration to the validation set to check for convergence. 
model.fit(datagen.flow(X_train, y_train, batch_size=8), epochs = 40, validation_data=(X_test, y_test), verbose = 1)


# save the model weights so we can use them for LIME later
model.save('trained_model')