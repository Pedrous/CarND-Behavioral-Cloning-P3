import os
import csv

# Open the new data that I have collected
samples = []
with open('./newdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

# Define a generator to save memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	for i in range(3):
            		# Define the image path from the .csv lines and load the image
		            name = './newdata/IMG/'+batch_sample[i].split('/')[-1]
		            center_image = cv2.imread(name)
		            # add or reduce the steering angle if it is a side camera image
		            if i == 1:
		            	center_angle = float(batch_sample[3]) + 0.2
		            if i == 2:
		            	center_angle = float(batch_sample[3]) - 0.2
		            else:
		            	center_angle = float(batch_sample[3])
		            # Extend the images and angles to be returned for the training
		            images.extend([center_image, cv2.flip(center_image,1)])
		            angles.extend([center_angle, -center_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

ch, row, col = 3, 160, 320  # Trimmed image format

# import the needed Keras tools
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

### THE MODEL ###
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation and the data is cropped
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(row, col, ch)))#, output_shape=(ch, row, col)))
model.add(Cropping2D(cropping=((70,25), (0,0))))#, input_shape = (row, col, ch)))

# First convolutional layer with dropout and activation
model.add(Convolution2D(24,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('elu'))

# Second convolutional layer with dropout and activation
model.add(Convolution2D(36,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('elu'))

# Third convolutional layer with dropout and activation
model.add(Convolution2D(48,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('elu'))

# Fourth convolutional layer with dropout and activation
model.add(Convolution2D(64,3,3))
model.add(Dropout(0.5))
model.add(Activation('elu'))

# Fifth convolutional layer with dropout and activation
model.add(Convolution2D(64,3,3))
model.add(Dropout(0.5))
model.add(Activation('elu'))

# Flattening
model.add(Flatten())#input_shape = (row, col, ch)))

# Four connected layers with activations
model.add(Dense(1164))
model.add(Activation('elu'))
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            6*len(train_samples), validation_data=validation_generator, \
            nb_val_samples=6*len(validation_samples), nb_epoch=6, verbose = 1)

# Save the model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
