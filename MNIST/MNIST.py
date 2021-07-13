# libs
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")


# Load dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


#Shape of dataset
# x_train(60,000 rows, 28 * 28 pixel config, 1 color channel)
# y_train(10,000 rows, 1 coloumn)
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)


#Normalize
x_train = x_train / 255
x_test = x_test / 255


#one_hot
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


############
#Model Arch#
############

model = Sequential()

#Layer 1 (convlution)
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)))

#Layer 2 (Pooling Layer)
model.add(MaxPooling2D(pool_size = (2,2)))

#Layer 3 (another convlution)
model.add(Conv2D(32, (3,3), activation = 'relu'))

#Layer 4 (pooling)
model.add(MaxPooling2D(pool_size = (2, 2)))

#layer 5 (flattening)
model.add(Flatten())

#1000 nueron
model.add(Dense(1000, activation = 'relu'))

#drop out layer
model.add(Dropout(0.5))

#500 nueron
model.add(Dense(500, activation = 'relu'))

#drop out layer
model.add(Dropout(0.5))

#250 nueron
model.add(Dense(250, activation = 'relu'))
# Last layer to give us classification
model.add(Dense(10, activation = 'softmax'))

###############
#Compile Model#
###############

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

 

#train model
hist = model.fit(x_train, y_train_one_hot,
                 batch_size = 256,
                 epochs = 10,
                 validation_split = 0.2)

##########
#Evaluate#
##########

model.evaluate(x_test, y_test_one_hot)[1]

