import os, shutil
import errno

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

"""

# Make an experimental training env.
ROOT_DIR = "D:/workspace/CHN_OCR/data_synthesis/"
DATA_DIR = "data/"
EXP_DIR = "experiment/"
# copy 5% of training/validating/testing set from "data/" to "experiment/"
class Utils(object):
    def __init__(self):
        pass
    def createExperimentEnv(source, dst):


    def shutil_copy(src, dest):
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
    u.createExperimentEnv(ROOT_DIR+DATA_DIR, ROOT_DIR+EXP_DIR)
