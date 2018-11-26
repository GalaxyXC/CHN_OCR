import os


from datetime import date, timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Dropout, Activation, ConvLSTM2D, Flatten, RNN, SimpleRNN
from keras.optimizers import Adam



ROOT_DIR = "data_synthesis"
DATA_DIR = "/data"

EPOCHS = 10
BATCH_SIZE = 128
NUM_CLASSES = len(os.listdir(ROOT_DIR + DATA_DIR + "/train/"))
RANDOM_SEED = 670


def train(keras_model,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE):
    # Load X and assign Y accordingly


    model = keras_model
    model.compile(loss='mean_square_error', optimizer='adam')
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size)
    score = model.evaluate(X_test, y_test,
                           batch_size=BATCH_SIZE*2)


def

def model_setup():
    # LeNet, CNN by Yan LeCun
    # network architect :   conv2d->max_pool2d ->
    #                       conv2d->max_pool2d ->
    #                       conv2d->max_pool2d ->
    #                       conv2d ->
    #                       conv2d->max_pool2d ->
    #                       fully_connected ->
    #                       fully_connected
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dense())
    return model

if __name__ == '__main__':
    model1 = model_setup()
    train(model1)


