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

# read metadata (image file locations and steering angles)
samples = []
with open(root_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Remove the labels from the CSV file for data ingestion.
# I kept the labels in the file itself for documentation.
samples = samples[1:]

# 80% training, 20% validation (not test in this case) split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# I recorded the training data several times in different directories so as not 
# to mess up any given dataset that I thought was giving good training results.
# I subsequently concatenated the datasets that were promising, but the concatenated 
# log file reflected these different directories in the image paths.
# For the final merger of datasets, this function allows us to ignore the prefix paths
# and normalize the path to the final merged dataset directory structure.
def normalize_path(path):
    return root_path + "/IMG/" + os.path.basename(path)

# Generate training samples. This function creates batches, produces synthetic
# angle offsets for left and right camera images, shuffles the data on each
# epoch, etc. 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)

        # generator serves one batch at a time
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            # Offset for left and right images to simulate a balanced
            # dataset rather than one with predominantly center driving.
            # This value is a tunable parameter. 
            angle_offset = 0.20 

            # Create examples for the batch (center, left and right images,
            # as well as mirror images of those).
            for batch_sample in batch_samples:

                # unpack the row from the metadata file.
                center_file, left_file, right_file, center_angle, *rest = tuple(batch_sample)

                # It's a CSV file so let's cast the string representation
                # of a the steering angle to a float.
                center_angle = float(center_angle)

                # Load all three images (center, left, right) for this metadata row.
                new_images = [cv2.imread(normalize_path(x)) for x in [center_file, left_file, right_file]]

                # Take center angles and compute synthetic left/right angles
                # based on the angle_offset parameter mentioned above. 
                new_angles = [center_angle, center_angle + angle_offset, center_angle - angle_offset]
      
                # Generate flipped images and angles. Include the center image for flipping
                # since the angle associated with it will often be non-zero as well.
                flipped_images = [np.fliplr(y) for y in new_images]
                flipped_angles = [-z for z in new_angles]

                # Add all images to the batch (center / left / right plus mirror images).
                images.extend(new_images)
                images.extend(flipped_images)
                 
                # Add angles corresponding to the above images to the batch.
                angles.extend(new_angles)
                angles.extend(flipped_angles)

            # Convert lists to NumPy arrays to be compatible with Keras.
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Use batch size of 16 for both training and validation. It's really 
# 16 * 6 = 96 images because we're including center / left / right
# images along with their mirror images.
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

# Original image dimensions to provide to first Keras layer.
ch, row, col = 3, 160, 320 

# NVIDIA model - see here: https://arxiv.org/pdf/1604.07316v1.pdf
# This is a nicer API than the one we used in the course.
# This Convolution2D layer already includes activation and subsampling parameters.
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
