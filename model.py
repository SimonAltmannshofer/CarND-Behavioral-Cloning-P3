import csv
import cv2
import numpy as np
import scipy.misc
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import h5py
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import toolbox

# define hyperparameters
epoch = 5
correction = 0.2
learning_rate = 1e-4
activation = 'elu'

# load the data
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
        
images = []
measurements = []
N = len(lines)
i_ex = random.randint(0,N)
print(i_ex)
i = 0

for line in lines:
    rand_img = random.randint(0,2)
    img_path = line[rand_img]
    img = cv2.imread(img_path)
    measurement = float(line[3])
    if rand_img == 1: # left image
        measurement += correction
    elif rand_img == 2: # right image
        measurement -= correction
        
    rand_flip = random.randint(0,1)
    if rand_flip == 1: # flip image
        img = np.fliplr(img)
        measurement = -measurement
    
    images.append(img)
    measurements.append(measurement)
    
    
plt.figure(0)
plt.subplot(1,2,1)
plt.imshow(img[60:135,:,::-1])    
plt.subplot(1,2,2)
plt.imshow(np.fliplr(img[60:135, :, ::-1]))



    
X_train = np.array(images)
y_train = np.array(measurements)

plt.figure(1)
plt.hist(measurements, bins = 500)
#plt.show()
   
print("Number of samples: ", len(y_train))
    


# train the data


model = Sequential()
# add an lambda layer to normalize the data
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
# add a cropping layer
model.add(Cropping2D(cropping=((60,25), (0,0))))



# nVIDIA network
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='same'))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='same'))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='same'))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='same'))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))


model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='same'))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Flatten())

#model.add(Dense(1164))
#model.add(Activation(activation))

model.add(Dense(100, activation=activation))
model.add(Activation(activation))
model.add(Dropout(rate=0.1))

model.add(Dense(50, activation=activation))
model.add(Activation(activation))

model.add(Dense(10, activation=activation))
model.add(Activation(activation))

model.add(Dense(1))

model.summary()

adam = Adam(lr=learning_rate)
model.compile(loss='mse', optimizer='adam')

start = timer()
history_object = model.fit(X_train, y_train, 
                           validation_split=0.2, 
                           shuffle=True, 
                           nb_epoch=epoch)
                           
print("Time for training [min]: ", (timer() - start)/60)

model.save('model.h5')
print("Training done!")

del model

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.figure(2)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

plt.show()