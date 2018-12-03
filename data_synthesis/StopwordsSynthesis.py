import os
import copy

import cv2
import numpy as np
from PIL import Image, ImageOps

#from DataSynthesis.DataSynthesis import *  # console
from DataSynthesis import *  # main

ROOT_DIR = "data_synthesis/"  # run in console
ROOT_DIR = ""  # Run in main()

# start of stopwords index
# NEXT_IDX = len(os.listdir(ROOT_DIR + "data/train/")) + 1

class StopwordsExpansion(object):
    def __init__(self, width, height,
                 margin,
                 output_dir):
        self.width = width
        self.height = height
        self.margin = margin
        self.output_dir = output_dir

    def img2char(self):
        pass

    def augment(self):
        pass

    def gen_img(self):
        pass


if __name__ == '__main__':
    data_expand = StopwordsExpansion(width=60,
                                    height=60,
                                    margin=4,
                                    output_dir="data/")
