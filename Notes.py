import os, shutil
import errno
import random



CHAR_RANGE = [0,3717]  # inclusive
EMOJI_RANGE = [3718,3897]  # inclusive


__notes__ = \
"""
MxNet:
Conv2D > MaxPooling > Relu Activation > Conv2D > AvgPooling > Relu Activation > Conv2D > AvgPooling > Relu Activation > Flatten > F.Connected * 4 > Concat > Softmax
( ref Also has LSTM+CTC )
https://blog.csdn.net/u013203733/article/details/79140499

Modified LeNets:
# network: conv2d->max_pool2d->conv2d->max_pool2d->conv2d->max_pool2d->conv2d->conv2d->max_pool2d->fully_connected->fully_connected
https://www.cnblogs.com/skyfsm/p/8443107.html#!comments

VGG16 ?
https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/

(multi-label classification)
smaller VGG
https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/

Model1:         Conv2D > MaxPool2D > Dense > Dropout > Dense=NUM_CLASSES;  epochs / time;           workers
Hyperpara.:     16(3,3)     /(2,2)      32      0.1              38         5       20min            1



"""


__ref__ = \
"""
CHN OCR, LeNet, Data Synthesis:
https://www.cnblogs.com/skyfsm/p/8443107.html#!comments
https://github.com/AstarLight/CPS-OCR-Engine/issues/33

OCR, captcha, K.GeneratorEnqueuer
http://ilovin.me/2017-04-06/tensorflow-lstm-ctc-ocr/

OCR variable length, CTC, GRU
https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/

CHN OCR, CNN / CTC+LSTM
https://blog.csdn.net/u013203733/article/details/79140499

OCR, K.FIFOQueue
https://dref360.github.io/input_pipeline/

TensorFlow multithreading
https://blog.metaflow.fr/tensorflow-how-to-optimise-your-input-pipeline-with-queues-and-multi-threading-e7c3874157e0

TensorFlow CNN Tutorial with MNIST Fashion
https://www.tensorflow.org/tutorials/keras/basic_classification

Keras Image processing kit
https://keras.io/preprocessing/image/#fit
"""

# Make an experimental training env.
ROOT_DIR = "D:/workspace/CHN_OCR/data_synthesis/"
DATA_DIR = "data/"
EXP_DIR = "experiment/"

NUM_CLASSES = len(os.listdir(ROOT_DIR + DATA_DIR + "train/"))


class Utils(object):
    def __init__(self):
        pass

    # copy 5% of training/validating/testing set from "data/" to "experiment/"
    def createExperimentEnv(self, src_dir, dst_dir):
        for i in range(NUM_CLASSES):
            if i % 100 == 0:
                print("copying... ", i)
                src = src_dir + "train/" + str(i).zfill(4)
                dst = dst_dir + "train/" + str(i).zfill(4)
                self.shutil_copy(src, dst)

                src = src_dir + "validate/" + str(i).zfill(4)
                dst = dst_dir + "validate/" + str(i).zfill(4)
                self.shutil_copy(src, dst)

                src = src_dir + "test/" + str(i).zfill(4)
                dst = dst_dir + "test/" + str(i).zfill(4)
                self.shutil_copy(src, dst)

    def generate_random_validation_set(self, src_dir, dst_dir, num_validation_samples):
        subdir = os.listdir(src_dir)
        num_experiment_classes = len(subdir)
        n = num_validation_samples

        labels = []

        while n > 0:
            dice = int(random.random() * num_experiment_classes)
            path = src_dir + subdir[dice]
            image_paths = os.listdir(path)
            labels.append(subdir[dice])


            dice2 = int(random.random() * len(image_paths))
            image_path = src_dir + subdir[dice] + "/" + image_paths[dice2]
            self.shutil_copy(image_path, dst_dir)

            n -= 1

        return labels


    def shutil_copy(self, src, dest):
        try:
            shutil.copytree(src, dest)
        except OSError as e:
            # If the error was caused because the source wasn't a directory
            if e.errno == errno.ENOTDIR:
                shutil.copy(src, dest)
            else:
                print('Directory not copied. Error: %s' % e)


if __name__ == '__main__':
    u = Utils()
    # u.createExperimentEnv(ROOT_DIR+DATA_DIR, ROOT_DIR+EXP_DIR)
    ll = u.generate_random_validation_set(ROOT_DIR+EXP_DIR+"validate/", ROOT_DIR+EXP_DIR+"validate1/", 8)
    print(ll)
    with open(ROOT_DIR+EXP_DIR+"validate1_labels.txt", 'w') as f:
        f.write(", ".join(ll))