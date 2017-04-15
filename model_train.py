import csv
import cv2
import numpy as np
import os
import sklearn

from keras.models import Model, Sequential
from keras.layers import Activation, Flatten, Dense, Dropout, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, Input, Merge
from keras.layers.core import Lambda
from keras import backend as K
from keras.optimizers import Adagrad, Adam
from keras.layers.convolutional import Convolution2D

import tensorflow as tf

from random import shuffle

from sklearn.model_selection import train_test_split

# data ingestion

root_path = './data/'

samples = []
with open(root_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Remove the labels from the CSV file
samples = samples[1:]

# 80% training, 20% validation (not test in this case)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# I recorded several times in different directories so as not to mess
# up any given dataset, and the concatenated log file reflected that.
# For the final merger of datasets, ignore the absolute paths
# from the various partial datasets.
def normalize_path(path):
    return root_path + "/IMG/" + os.path.basename(path)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            # offset for left and right images to simulate a balanced
            # rather than just learn center driving.
            angle_offset = 0.20 

            for batch_sample in batch_samples:

                # unpack the row
                center_file, left_file, right_file, center_angle, *rest = tuple(batch_sample)

                # it's a CSV file so let's cast the string representation
                # of a float to an actual float
                center_angle = float(center_angle)

                # load all the images
                new_images = [cv2.imread(normalize_path(x)) for x in [center_file, left_file, right_file]]

                # take center angles and compute
                # synthetic left/right angles
                new_angles = [center_angle, center_angle + angle_offset, center_angle - angle_offset]
      
                # generate flipped images and angles
                flipped_images = [np.fliplr(y) for y in new_images]
                flipped_angles = [-z for z in new_angles]

                # add images to the batch
                images.extend(new_images)
                images.extend(flipped_images)
                 
                # add angles to the batch
                angles.extend(new_angles)
                angles.extend(flipped_angles)

            # convert lists to NumPy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# use batch size of 16 for both training and validation
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

# original image dimensions
ch, row, col = 3, 160, 320 

# NVIDIA model - see here: https://arxiv.org/pdf/1604.07316v1.pdf

# This is a nicer API than the one we user in the course.
# This convolution layer already includes activation and subsampling
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1))) 
model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

# Use Adam with standard hyperparameters, but expose them 
# so they can be tweaked if need be.
opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)
model.compile(loss='mse', optimizer=opt)

# Print the layers and their dimensions.
print(model.summary())

# We multiply the number of training samples by 6 to account for 
# center, left and right images, and their flipped counterparts.
model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*6, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
