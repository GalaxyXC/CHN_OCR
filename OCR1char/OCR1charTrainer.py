import os
import time
import pickle

import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, Reshape
from keras.preprocessing import image

ROOT_DIR = "D:/workspace/CHN_OCR/"
DATA_DIR = "data_synthesis/experiment/"
MODEL_DIR = "OCR1char/"

TRAIN_DIR = ROOT_DIR + DATA_DIR + "train/"
VALIDATE_DIR = ROOT_DIR + DATA_DIR + "validate/"

EPOCHS = 5
BATCH_SIZE = 32
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
        # Import training file names
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
        # Import validating file names
        with open(MODEL_DIR + "validating_file_names.pkl", 'rb') as f:
            X_validate, y_validate = pickle.load(f)

        # use Keras.preprocessing.image.ImageDataGenerator()
        train_datagen = image.ImageDataGenerator()
        tik = time.time()
        train_gen = train_datagen.flow_from_directory(TRAIN_DIR,
                                                      target_size=(60, 60),
                                                      color_mode='grayscale',
                                                      seed=RANDOM_SEED,
                                                      batch_size=BATCH_SIZE,
                                                      class_mode='sparse')
        class_dict = train_gen.class_indices

        print("Deploy data generator: ", time.time() - tik)

        model = keras_model
        # one-hot encoded: use 'categorical_crossentropy' as loss, otherwise use 'sparse...'
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam')

        tik = time.time()
        model.fit_generator(train_gen,
                            steps_per_epoch=len(X_train) // BATCH_SIZE,
                            epochs=EPOCHS)
        print("Model fitting total time: ", time.time() - tik)
        return model, class_dict

    # Deprecated
    def tfdata_generator(self, images, labels,
                         train=True,
                         batch_size=BATCH_SIZE):
        """
        :param images:  tf.string, path/to/images_files
        :param labels: tf.int32,
        :param train: Boolean, True = training data, False = Validating/testing data
        :param batch_size: int, default
        :return: tf.Dataset generator for pipeline
        """

        # Construct a data generator using tf.Dataset
        def preprocess(image, label):
            # Preprocess raw data into trainable input.
            image_array = tf.image.decode_png(tf.read_file(image), channels=1)
            # Convert to float values in [0, 1]
            img = tf.image.convert_image_dtype(image_array, tf.float32)
            x = tf.reshape(img, (100, 100, 1))
            y = tf.one_hot(tf.convert_to_tensor(label), depth=NUM_CLASSES)
            return x, y

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        if train:
            dataset = dataset.shuffle(len(images))  # depends on sample size
        dataset = dataset.map(preprocess, num_parallel_calls=4).batch(batch_size).repeat()
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        iterator = dataset.make_one_shot_iterator()

        next_batch = iterator.get_next()
        while True:
            yield K.get_session().run(next_batch)
        #return dataset

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
        # Simple NN
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(60, 60, 1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        #model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        #model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        return model


if __name__ == '__main__':
    tn = OCR1charTrainer()
    model1 = tn.model1_setup() # simple NN

    trained, class_dict = tn.train(model1, epochs=20)

    """    
    #DEV CODES
    # RUN import
    # config constants
    # run pickle.load(training_files)
    """
    with open(ROOT_DIR+"experiment/validate1_labels.txt", 'r') as f:
        y_validate = f.read().split(", ")
    n = len(y_validate)

    X_filenames = os.listdir(ROOT_DIR + "experiment/validate1/")
    for i in range(n):
        i = 1
        x_filename = ROOT_DIR + "experiment/validate1/" + X_filenames[i]
        x_data = cv2.imread(x_filename, 0) # read as grayscale
        x_data = tf.reshape(x_data, (1,60,60,1))

        y_truth = y_validate[i]
        y_pred_onehot_encode = trained.predict(x_data, steps=1).tolist()[0]
        y_pred = class_dict[y_pred_onehot_encode.index(1.0)]

        print("truth: ", y_truth, " prediction: ", y_pred)



