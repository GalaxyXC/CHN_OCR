import os

import cv2
import numpy as np

import data_synthesis.data_synthesis as syn

ROOT_DIR = "data_synthesis/" # run in console
#ROOT_DIR = syn.ROOT_DIR  # Running main()

SOURCE_DIR = "emoji/wechat/source/"


class DataExpansion(object):
    def __init__(self, width, height,
                 margin,
                 output_dir):
        self.width = width
        self.height = height
        self.margin = margin
        self.output_dir = output_dir

    def expand_emojis(self):
        # crop emoji's from source
        cropped = []
        source = os.listdir(ROOT_DIR + SOURCE_DIR)

        for sc in source:
            im = cv2.imread(ROOT_DIR + SOURCE_DIR + sc, 0)
            # crop upper-left emoji [1200,10,125]
            r_delta = 125
            c_delta = 154
            for i in range(3):
                for j in range(7):
                    if i==2 and j==6:
                        continue  # delete key symbol

                    r = 1190 + r_delta * i
                    c = 0 + c_delta * j
                    emo = im[r:r + r_delta, c:c + c_delta]
                    # contrast streching
                    _min = np.amin(emo)
                    _max = np.amax(emo)
                    for r in range(emo.shape[0]):
                        for c in range(emo.shape[1]):
                            emo[r][c] = (emo[r][c] - _min)*255 // (_max - _min)
                    cropped.append(emo)


        emoji_list = []


    def expand_stopwords(self):
        pass

if __name__ == '__main__':
    data_expand = DataExpansion(width=60,
                                height=60,
                                margin=4,
                                output_dir="experiment")
    """
    # cv2 display image code:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', cropped[20])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
