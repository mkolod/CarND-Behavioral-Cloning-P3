import csv
import cv2
import numpy as np
import sklearn

from keras.models import Model, Sequential
from keras.layers import Activation, Flatten, Dense, Dropout, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, Input, Merge
from keras.layers.core import Lambda
from keras import backend as K
from keras.optimizers import Adagrad, Adam

from common import crop, resize

import tensorflow as tf

from random import choice, shuffle

from sklearn.model_selection import train_test_split

# data ingestion

root_path = './data/'

samples = []
with open(root_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Remove the labels
samples = samples[1:]

#lines = []

#with open('./data/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)

#lines = lines[1:]

#images = []
#steering_angles = []

#for line in lines:
#    # center image
#    source_path = line[0]
#    # extract file name and add it to another path
#    filename = source_path.split('/')[-1]
#    current_path = './data/IMG/' + filename
#    image = cv2.imread(current_path)
#    images.append(image)
#    steering_angle = float(line[3])
#    steering_angles.append(steering_angle)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def random_translation(image, max_pixels=10, angle_shift_per_pixel=0.004):
    shift = np.random.uniform(-max_pixels, max_pixels)
    rows, cols, _ = np.shape(image)
    trans = cv2.warpAffine(
        image,
        np.float32([[1, 0, shift], [0, 1, 0]]),
        (cols, rows)
    )
    shift = angle_shift_per_pixel * shift
    return trans, shift

def is_flip():
    return np.random.choice([True, False])

def flip(image, angle):
    return np.fliplr(image), -angle    

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            angle_offset = 0.20 

            for batch_sample in batch_samples:
               

                center_name = root_path + 'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
 
                if is_flip():
                    center_image, center_angle = flip(center_image, center_angle)
         
#                center_image, angle_shift = random_translation(center_image)
#                center_angle -= angle_shift

                left_name =  root_path + 'IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_angle = center_angle + angle_offset
#                left_image, angle_shift = random_translation(left_image)
#                left_angle -= angle_shift

                if is_flip():
                    left_image, left_angle = flip(left_image, left_angle)

                right_name =  root_path + 'IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_angle = center_angle - angle_offset
#                right_image, angle_shift = random_translation(right_image)
#                right_angle -= angle_shift

                if is_flip():
                    right_image, right_angle = flip(right_image, right_angle)

#                center_image = crop(center_image)
#                left_image = crop(left_image)
#                right_image = crop(right_image)
       
#                center_image = resize(center_image)
#                left_image = resize(left_image)
#                right_image = resize(right_image)

                images.append(center_image)
                angles.append(center_angle)
              
#                images.append(np.fliplr(center_image))
#                angles.append(-center_angle)
                   
                images.append(left_image)
                angles.append(left_angle)

#                images.append(np.fliplr(left_image))
#                angles.append(-left_angle)
                
                images.append(right_image)
                angles.append(right_angle)

#                images.append(np.fliplr(right_image))
#                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

ch, row, col = 3, 160, 320 # 160, 320 #3, 90, 300

# X_train = np.array(images[:4000])
#y_train = np.array(steering_angles[:4000])

# NOTE: save model in h5 format or dump NumPy arrays

# build model

#model = Sequential()

#model.add(Lambda(lambda x: x/127.5 - 1.,
#        input_shape=(row, col, ch),
#        output_shape=(row, col, ch)))

# model.add(Cropping2D(cropping=((50, 20), (10, 10))))

#model.add(Conv2D(32, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(32, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(64, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(128, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(128, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(512, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(256, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(Conv2D(256, 3, 3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Flatten())
# model.add(Dropout(0.5))
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1))

#model.add(Flatten(input_shape=[row, col, ch]))
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

from keras.layers.convolutional import Convolution2D

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
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

#model = to_multi_gpu(model, n_gpus=2)
# opt = Adagrad(lr=0.001)
opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)
model.compile(loss='mse', optimizer=opt)
# model.compile(loss='mse', optimizer='adam')

print(model.summary())

model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*3, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
