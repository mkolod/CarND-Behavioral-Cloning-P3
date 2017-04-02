import csv
import cv2
import numpy as np
import sklearn

from keras.models import Model, Sequential
from keras.layers import Activation, Flatten, Dense, Dropout, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, Input, merge
from keras.layers.core import Lambda
from keras import backend as K

import tensorflow as tf

from random import choice, shuffle

from sklearn.model_selection import train_test_split

# data ingestion

samples = []
with open('./sample_data/driving_log.csv') as csvfile:
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

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)

#        batch_size /= 2
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            angle_offset = 0.08

            for batch_sample in batch_samples:
               
#                print(batch_sample[0:3])

                center_name = './sample_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                
                left_name =  './sample_data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_angle = center_angle + angle_offset  

                right_name =  './sample_data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_angle = center_angle - angle_offset

#                if choice([True, False]):
#                    center_image = np.fliplr(center_image)
#                    center_angle = -center_angle
                images.append(center_image)
                angles.append(center_angle)
              
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                   
                images.append(left_image)
                angles.append(left_angle)

                images.append(np.fliplr(left_image))
                angles.append(-left_angle)
                
                images.append(right_image)
                angles.append(right_angle)

                images.append(np.fliplr(right_image))
                angles.append(-right_angle)

#                images.append(np.fliplr(center_image))
#                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

ch, row, col = 3, 160, 320

# X_train = np.array(images[:4000])
#y_train = np.array(steering_angles[:4000])

# NOTE: save model in h5 format or dump NumPy arrays

# build model

model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=[row, col, ch]))
model.add(Conv2D(16, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def to_multi_gpu(model, n_gpus=2):
    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name=model.inputs[0]) #_names[0])

    towers = []

    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape,
                arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))


    with tf.device('/cpu:0'):
        merged = merge(towers, mode='concat', concat_axis=0)

    return Model(input=[x], output=merged)

#model.add(Flatten(input_shape=[row, col, ch]))
#model.add(Dense(128))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))

#model = to_multi_gpu(model, n_gpus=2)
model.compile(loss='mse', optimizer='adam')

print(model.summary())

model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples)*6, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model.h5')
