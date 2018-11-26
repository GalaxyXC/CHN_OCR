import os
from datetime import date, timedelta
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, Conv2D, Flatten, MaxPool2D
from keras.optimizers import Adam

ROOT_DIR = "data_synthesis/"
DATA_DIR = "data/"
TRAIN_DIR = ROOT_DIR + DATA_DIR + "train/"
VALIDATE_DIR = ROOT_DIR + DATA_DIR + "validate/"
MODEL_DIR = "OCR1char/"

EPOCHS = 10
BATCH_SIZE = 128

NUM_CLASSES = len(os.listdir(TRAIN_DIR))

RANDOM_SEED = 670

class OCR1charTrainer(object):
    def __init__(self):
        pass

    def train(self, keras_model,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE):

        # Load X and assign Y accordingly
        labels = [i+1 for i in range(NUM_CLASSES)]

        # Generate Training file names
        """
        X_train = [""] * (NUM_CLASSES * len(os.listdir(TRAIN_DIR + '0001/'))) # list of file names
        y_train = [0] * (NUM_CLASSES * len(os.listdir(TRAIN_DIR + '0001/'))) # list of labels

        idx = 0
        for id_label in labels:
            print(id_label)
            folder = TRAIN_DIR + str(id_label).zfill(4) + "/"
            file_names = os.listdir(folder)

            try:
                X_train[idx:idx+len(file_names)] = [folder + file for file in file_names]
                y_train[idx:idx+len(file_names)] = [id_label] * len(file_names)
                idx += len(file_names)
            except Exception as e:
                print(e)
                X_train[idx:] = [folder + file for file in file_names[X_train[idx:]:]]
                y_train[idx:] = [id_label] * len(y_train[idx:])

        with open(MODEL_DIR + "training_file_names.pkl", 'wb') as f:
            pickle.dump([X_train, y_train], f)
        """
        with open(MODEL_DIR + "training_file_names.pkl", 'rb') as f:
            X_train, y_train = pickle.load(f)

        # Generate Validating file names
        """
        X_validate = [""] * (NUM_CLASSES * len(os.listdir(VALIDATE_DIR + '0001/'))) # list of file names
        y_validate = [0] * (NUM_CLASSES * len(os.listdir(VALIDATE_DIR + '0001/'))) # list of labels

        idx = 0
        for id_label in labels:
            print(id_label)
            folder = VALIDATE_DIR + str(id_label).zfill(4) + "/"
            file_names = os.listdir(folder)

            try:
                X_validate[idx:idx+len(file_names)] = [folder + file for file in file_names]
                y_validate[idx:idx+len(file_names)] = [id_label] * len(file_names)
                idx += len(file_names)
            except Exception as e:
                print(e)
                X_validate[idx:] = [folder + file for file in file_names[X_validate[idx:]:]]
                y_validate[idx:] = [id_label] * len(y_validate[idx:])

        with open(MODEL_DIR + "validating_file_names.pkl", 'wb') as f:
            pickle.dump([X_validate, y_validate], f)
        """
        with open(MODEL_DIR + "validating_file_names.pkl", 'rb') as f:
            X_validate, y_validate = pickle.load(f)

        training_set = self.tfdata_generator(X_train, y_train, train=True)
        validating_set = self.tfdata_generator(X_validate, y_validate, train=False)

        model = keras_model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size)

        score = model.evaluate(X_validate, y_validate, batch_size=batch_size*2)
        return score

    def tfdata_generator(self, images, labels,
                         train=True,
                         batch_size=BATCH_SIZE):
        # Construct a data generator using tf.Dataset
        def preprocess(image, label):
            # Preprocess raw data into trainable input.
            imgs = []
            x = tf.reshape(tf.cast(image, tf.float32), (100, 100, 1))
            y = tf.one_hot(tf.cast(label, tf.uint8), NUM_CLASSES)
            return x, y

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        if train:
            dataset = dataset.shuffle(1000)  # depends on sample size

        # Transform and batch data at the same time
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            preprocess, batch_size,
            num_parallel_batches=4,  # cpu cores
            drop_remainder=True if train else False))
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    def model2_setup(self):
        # LeNet, CNN by Yan LeCun
        # network architect :   conv2d->max_pool2d ->
        #                       conv2d->max_pool2d ->
        #                       conv2d->max_pool2d ->
        #                       conv2d ->
        #                       conv2d->max_pool2d ->
        #                       fully_connected ->
        #                       fully_connected
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dense(128, activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        return model

    def model1_setup(self):
        # Simple DNN
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dense(128, activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        return model

if __name__ == '__main__':
    tn = OCR1charTrainer()
    model1 = tn.model_setup() # simple DNN
    tn.train(model1)


