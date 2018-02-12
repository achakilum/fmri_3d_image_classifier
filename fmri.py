# chakilu2
# CS 446

from __future__ import print_function

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation, Conv3D, MaxPooling3D, ZeroPadding3D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

print("Initializing variables...")

batch_size = 32
fmri_img_shape = (26, 31, 23, 1)
num_train_samples = 4602
num_classes = 19
num_test_samples = 1971
epochs = 200

print("Loading data...")

train_x = np.load('train_X.npy')
train_x = train_x.astype('float32')
train_x = np.reshape(train_x, (num_train_samples, 26, 31, 23, 1))

train_y = np.load('train_binary_Y.npy')

print("Implementing model...")

model = Sequential()

model.add(ZeroPadding3D((1,1,1),input_shape=fmri_img_shape))
model.add(Conv3D(64, kernel_size=(3,3,3), activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Conv3D(64, kernel_size=(3,3,3), activation='relu'))
model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

model.add(ZeroPadding3D((1,1,1)))
model.add(Conv3D(128, kernel_size=(3,3,3), activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Conv3D(128, kernel_size=(3,3,3), activation='relu'))
model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

model.add(ZeroPadding3D((1,1,1)))
model.add(Conv3D(256, kernel_size=(3,3,3), activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Conv3D(256, kernel_size=(3,3,3), activation='relu'))
model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

print("Compiling model...")

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
			  
print("Fitting model...")

early_stopper = EarlyStopping(patience=2)
history = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
					callbacks=[early_stopper],
                    verbose=1)

print("Predicting...")

test_x = np.load('valid_test_X.npy')
test_x = test_x.astype('float32')
test_x = np.reshape(test_x, (num_test_samples, 26, 31, 23, 1))
					
predictions = model.predict(test_x, 
							batch_size=batch_size, 
							verbose=1)
							
np.save('output_test_Y.npy', predictions)

predictions[predictions>=0.5] = 1
predictions[predictions<0.5] = 0

np.save('output_test_Y_binary.npy', predictions)
