import os
import time
import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, MaxPool2D
from keras.optimizers import Adam

ROOT_DIR = "D:/workspace/CHN_OCR/data_synthesis/"
DATA_DIR = "data/"
TRAIN_DIR = ROOT_DIR + DATA_DIR + "train/"
VALIDATE_DIR = ROOT_DIR + DATA_DIR + "validate/"
MODEL_DIR = "OCR1char/"

EPOCHS = 10
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
        with open("training_file_names.pkl", 'rb') as f:
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
        with open("validating_file_names.pkl", 'rb') as f:
            X_validate, y_validate = pickle.load(f)

        model = keras_model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        tik = time.time()
        """
        model.fit(xtr, ytr,
                  steps_per_epoch=len(X_train) // BATCH_SIZE,
                  epochs=epochs,
                  batch_size=batch_size)
        """
        model.fit_generator(generator=self.tfdata_generator(X_train, y_train, train=True),
                            steps_per_epoch=len(X_train) // batch_size,
                            workers=0)
        print(time.time() - tik)

        # score = model.evaluate(validating_set.make_one_shot_iterator(), batch_size=batch_size*2)
        return

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
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 1)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        #model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        #model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dense(64, activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        return model

if __name__ == '__main__':
    tn = OCR1charTrainer()
    model1 = tn.model1_setup() # simple DNN

    print("model set.")
    tn.train(model1, epochs=5)


    """
    [Error message]
    
    ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[307328,3817] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu
	 [[node training/Adam/Variable_4/Assign (defined at D:\workspace\CHN_OCR\venv\lib\site-packages\keras\backend\tensorflow_backend.py:402)  = Assign[T=DT_FLOAT, _grappler_relax_allocator_constraints=true, use_locking=true, validate_shape=true, _device="/job:localhost/replica:0/task:0/device:CPU:0"](training/Adam/Variable_4, training/Adam/zeros_10)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
    
    
    #DEV CODES
    # RUN import
    # config constants
    # run pickle.load(training_files)
    
    X_train, y_train = X_train[:1000], y_train[:1000]
    images, labels = X_train, y_train 
    batch_size = BATCH_SIZE
    train = True
    
    """

