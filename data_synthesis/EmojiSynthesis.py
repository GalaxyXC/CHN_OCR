import os
import copy

import cv2
import numpy as np
from scipy import stats
from PIL import Image, ImageOps

#from DataSynthesis.DataSynthesis import *  # console
from DataSynthesis import *  # main

ROOT_DIR = "data_synthesis/"  # run in console
ROOT_DIR = ""  # Run in main()
EMOJI_DIR = "emoji/"

# start of emoji index
# NEXT_IDX = len(os.listdir(ROOT_DIR + "data/train/")) + 1
NEXT_IDX = 3817 + 1


class EmojiSynthesis(object):
    def __init__(self, width, height,
                 margin,
                 output_dir):
        self.width = width
        self.height = height
        self.margin = margin
        self.output_dir = output_dir

    def crop_emoji(self, source, platform):
        # crop emoji's from source
        cropped = []
        files_dir = source + platform + "/"
        files = os.listdir(files_dir)

        for sc in files:
            im = cv2.imread(files_dir + sc, 0)
            r_delta = 125
            c_delta = 154
            for i in range(3):
                for j in range(7):
                    if i==2 and j==6:
                        continue  # delete key symbol

                    r = 1190 + r_delta * i
                    c = 0 + c_delta * j
                    emo = im[r:r + r_delta, c:c + r_delta]
                    # contrast streching
                    thres = stats.mode(emo, axis=None)  # set max value = mode of ndarray

                    _min = np.amin(emo)
                    _max = int(thres[0])

                    _, emo = cv2.threshold(emo, _max, 255, cv2.THRESH_TRUNC)

                    for r in range(emo.shape[0]):
                        for c in range(emo.shape[1]):
                            emo[r][c] = (emo[r][c] - _min)*255 // (_max - _min)
                    cropped.append(emo)
        return cropped

    def augment_emoji(self, emoji_list):
        """
        :param self:
        :param emoji_list: list of cv2.image (=numpy.ndarray)
        :return:  list of augmented emoji's
        """
        np_img_list = []
        proc = Processor()
        for emoji in emoji_list:
            pil = Image.fromarray(emoji)
            for angle in [-15, -10, -5, 5, 10, 15]:
                pil_rotate = ImageOps.invert(pil).rotate(angle)
                np_img = np.asarray(pil_rotate, dtype='uint8')
                #np_img = np_img.reshape((self.height, self.width))
                # crop empty margins
                cropped_box = proc.find_image_bbox(np_img)
                left, upper, right, lower = cropped_box
                np_img = np_img[upper: lower + 1, left: right + 1]
                # Resize, keep ratio, fill background
                resizer = PreprocessResizeKeepRatioFillBG(self.width, self.height,
                                                          fill_bg=False,
                                                          margin=self.margin)
                np_img = resizer.do(np_img)
                np_img_list.append(np_img)

        if not np_img_list:
            return []

        #   Augment image:
        aug = Augmentor()
        #   pepper-salt noise, erosion, dilation
        noised, eroded, dilated = [], [], []
        for each in copy.deepcopy(np_img_list):
            augmented = aug.add_noise(each)
            noised.append(augmented)
        for each in copy.deepcopy(np_img_list):
            augmented = aug.add_erode(each)
            eroded.append(augmented)
        for each in copy.deepcopy(np_img_list):
            augmented = aug.add_dilate(each)
            dilated.append(augmented)

        return np_img_list + noised + eroded + dilated

    def gen_img(self, platforms,
                origin_emoji_list,
                max_iter=-1,
                next_idx=0):
        if max_iter<0:
            max_iter = len(origin_emoji_list)

        iteration = 0  # = number of chars to gen. image
        for emoji in origin_emoji_list:
            print(iteration)
            for platform in platforms:
                image_list = self.augment_emoji([emoji])
                gen = DataGenerator(60,60,4,self.output_dir)
                gen.split_test_train_write(image_list, str(iteration+next_idx), font_name=platform)

            iteration += 1
            if iteration >= max_iter:
                break

    def expand_stopwords(self):
        pass

if __name__ == '__main__':
    data_expand = EmojiSynthesis(width=60,
                                 height=60,
                                 margin=4,
                                 output_dir="data/")

    platforms = ["wechat"]
    # crop image
    cropped = []
    for p in platforms:
        cropped += data_expand.crop_emoji(ROOT_DIR + EMOJI_DIR, p)

    # map from id to emoji: save origin cropped emojis into disk

    for i in range(len(cropped)):
        idx = str(NEXT_IDX + i).zfill(4)
        cv2.imwrite(ROOT_DIR + EMOJI_DIR + "mapping/" + idx + ".png", cropped[i])

    # augment file and write augmented data into disk, split train/validate/test
    data_expand.gen_img(platforms,
                        cropped,
                        max_iter=-1,
                        next_idx=NEXT_IDX)


    """
    # cv2 display image code:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', emo2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
