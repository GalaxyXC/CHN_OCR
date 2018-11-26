import os
import copy
import pickle
import argparse
import cv2
import random
import time

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Set ROOT_DIR to "data_synthesis/" when run in PyCharm's Console, Set to "" otherwise
ROOT_DIR = "data_synthesis/"
ROOT_DIR = ""
FONT_DIR = 'fonts/'
OUTPUT_DIR = 'data/'

GB2312_FILE = "GB2312-UTF8_2.txt"
FONT0 = 'msyh.ttc'

HEIGHT = 100
WIDTH = 100
MARGIN = 4

TRAIN_VALIDATE_TEST_RATIO = [70,20,10]

class Processor(object):
    def __init__(self, ):
        pass

    def find_image_bbox(self, img):
        # Find minimum bounding box
        height = img.shape[0]
        width = img.shape[1]
        v_sum = np.sum(img, axis=0)
        h_sum = np.sum(img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1
        # Scan left to right
        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break
        # Scan right to left
        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break
        # Scan top-down
        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break
        # Scan bottom-up
        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break
        return (left, top, right, low)

    def preprocess_resize_keep_ratio(self, cv2_img, width, height):
        cur_height, cur_width = cv2_img.shape[:2]

        ratio_w = float(width) / float(cur_width)
        ratio_h = float(height) / float(cur_height)
        ratio = min(ratio_w, ratio_h)

        new_size = (min(int(cur_width * ratio), width),
                    min(int(cur_height * ratio), height))

        new_size = (max(new_size[0], 1),
                    max(new_size[1], 1),)

        resized_img = cv2.resize(cv2_img, new_size)
        return resized_img


# Resize while keep ratio
class PreprocessResizeKeepRatioFillBG(object):

    def __init__(self, width, height,
                 fill_bg=False,
                 auto_avoid_fill_bg=True,
                 margin=None):
        self.width = width
        self.height = height
        self.fill_bg = fill_bg
        self.auto_avoid_fill_bg = auto_avoid_fill_bg
        self.margin = margin

    @classmethod
    def is_need_fill_bg(cls, cv2_img, th=0.5, max_val=255):
        image_shape = cv2_img.shape
        height, width = image_shape
        if height * 3 < width:
            return True
        if width * 3 < height:
            return True
        return False

    @classmethod
    def put_img_into_center(cls, img_large, img_small, ):
        width_large = img_large.shape[1]
        height_large = img_large.shape[0]

        width_small = img_small.shape[1]
        height_small = img_small.shape[0]

        if width_large < width_small:
            raise ValueError("width_large <= width_small")
        if height_large < height_small:
            raise ValueError("height_large <= height_small")

        start_width = (width_large - width_small) // 2
        start_height = (height_large - height_small) // 2

        img_large[start_height:start_height + height_small,
                  start_width:start_width + width_small] = img_small
        return img_large

    def do(self, cv2_img):
        # 确定有效字体区域，原图减去边缘长度就是字体的区域
        if self.margin is not None:
            width_minus_margin = max(2, self.width - self.margin)
            height_minus_margin = max(2, self.height - self.margin)
        else:
            width_minus_margin = self.width
            height_minus_margin = self.height

        cur_height, cur_width = cv2_img.shape[:2]
        if len(cv2_img.shape) > 2:
            pix_dim = cv2_img.shape[2]
        else:
            pix_dim = None

        proc = Processor()
        resized_cv2_img = proc.preprocess_resize_keep_ratio(cv2_img, width_minus_margin, height_minus_margin)

        if self.auto_avoid_fill_bg:
            need_fill_bg = self.is_need_fill_bg(cv2_img)
            if not need_fill_bg:
                self.fill_bg = False
            else:
                self.fill_bg = True

        ## should skip horizontal stroke
        if not self.fill_bg:
            ret_img = cv2.resize(resized_cv2_img, (width_minus_margin,
                                                   height_minus_margin))
        else:
            if pix_dim is not None:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin),
                                    np.uint8)
            # put resized image into center
            ret_img = self.put_img_into_center(norm_img, resized_cv2_img)

        if self.margin is not None:
            if pix_dim is not None:
                norm_img = np.zeros((self.height,
                                     self.width,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((self.height,
                                     self.width),
                                    np.uint8)
            ret_img = self.put_img_into_center(norm_img, ret_img)
        return ret_img


def args_parse():
    """
    Command line:
    python gen_printed_char.py --out_dir ./dataset --font_dir ./chinese_fonts --width 30 --height 30 --margin 4 --rotate 30 --rotate_step 1
    --out_dir: output directory
    --font_dir: font file
    --width, --height, --margin: output size and margin
    --rotate: rotate from -$rotate to +$rotate, by $rotate_step
    """
    parser = argparse.ArgumentParser(
        description="CHN OCR DATA SYNTHESIS", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out_dir', dest='out_dir',
                        default=None, required=True,
                        help='write a caffe dir')
    parser.add_argument('--font_dir', dest='font_dir',
                        default=None, required=True,
                        help='font dir to to produce images')
    parser.add_argument('--test_ratio', dest='test_ratio',
                        default=0.2, required=False,
                        help='test dataset size')
    parser.add_argument('--width', dest='width',
                        default=None, required=True,
                        help='width')
    parser.add_argument('--height', dest='height',
                        default=None, required=True,
                        help='height')
    parser.add_argument('--no_crop', dest='no_crop',
                        default=True, required=False,
                        help='', action='store_true')
    parser.add_argument('--margin', dest='margin',
                        default=0, required=False,
                        help='', )
    parser.add_argument('--rotate', dest='rotate',
                        default=0, required=False,
                        help='max rotate degree 0-45')
    parser.add_argument('--rotate_step', dest='rotate_step',
                        default=0, required=False,
                        help='rotate step for the rotate angle')
    parser.add_argument('--need_aug', dest='need_aug',
                        default=False, required=False,
                        help='need data augmentation', action='store_true')
    args = vars(parser.parse_args())
    return args


class Augmentor(object):
    def __init__(self):
        pass

    def add_noise(self, img, noise_count=20):
        for i in range(noise_count):  # salt & pepper noise for binary image
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 255 - img[temp_x][temp_y]
        return img

    def add_erode(self, img, kernel_size=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.erode(img, kernel)

    def add_dilate(self, img, kernel_size=3):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.dilate(img, kernel)


class DataGenerator(object):
    def __init__(self):
        pass

    def map_id_char(self, char_file_path, map_backup_name=""):
        # read all chars in GB2312 txt file into memory
        all_char = ""
        with open(file=char_file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                all_char += line

        discard_chars = "ABCDEFo\n0123456789abcdef+ "
        all_char = [e for e in all_char if e not in discard_chars][1:]
        # print(len(all_char)) # 3755
        all_char += "1234567890"
        all_char += "".join([chr(ord('a') + i) for i in range(26)])
        all_char += "".join([chr(ord('A') + i) for i in range(26)])

        # create dictionary dict[id]=char
        mapping_idx_to_char = {}
        # mapping_char_to_idx = {}
        for idx in range(len(all_char)):
            key = str(idx + 1).zfill(4)
            val = all_char[idx]  # .encode('utf-8')
            mapping_idx_to_char[key] = val
            # mapping_char_to_idx[val] = key
        if map_backup_name:
            with open(map_backup_name + ".pkl", 'wb') as f:
                pickle.dump(mapping_idx_to_char, f)

        """
        # retrieve mappings
            with open(DATA_DIR + "mapping_idx_to_char.pkl", 'rb') as f:
            tmp = pickle.load(f)

        # reverse dictionary
            pickle_map_char_to_idx =  "mapping_char_to_idx.pkl" 
            with open(char_path, 'wb') as f:
                pickle.dump(mapping_char_to_idx, f)
        """
        # output the name of stored pickles
        return mapping_idx_to_char

    def char2img(self, idx, char, font):
        image_list = []
        # generate plain image
        # black background
        img = Image.new("RGB", (WIDTH, HEIGHT), "black")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font, int(WIDTH * 0.7), )
        # white char.
        draw.text((0, 0), char, (255, 255, 255), font=font)
        image_list.append(copy.deepcopy(img))

        # data augmentation:
        #   gen. rotate image (-15, -10, 5, 0, 5, 10, 15)
        for angle in [-15, -10, -5, 5, 10, 15]:
            img_rotate = img.rotate(angle)
            image_list.append(img_rotate)

        # retrieve 3-channel numpy matrix for each image in image_list
        np_img_list = []
        proc = Processor()
        for each in image_list:
            img_data = list(each.getdata())
            sum_val = 0
            for i in img_data:
                sum_val += sum(i)

            if sum_val > 2:
                np_img = np.asarray(img_data, dtype='uint8')
                np_img = np_img[:, 0]
                np_img = np_img.reshape((HEIGHT, WIDTH))
                # crop empty margins

                cropped_box = proc.find_image_bbox(np_img)
                left, upper, right, lower = cropped_box
                np_img = np_img[upper: lower + 1, left: right + 1]
                # Resize, keep ratio, fill background
                resizer = PreprocessResizeKeepRatioFillBG(WIDTH, HEIGHT, fill_bg=False, margin=MARGIN)
                np_img = resizer.do(np_img)

                np_img_list.append(np_img)
            else:
                print("no numpy-image found. ", idx)

        if not np_img_list:
            return []

        #   Augment image:
        aug = Augmentor()
        noised, eroded, dilated = [], [], []
        for each in copy.deepcopy(np_img_list):
            #   pepper-salt noise
            img_aug = aug.add_noise(each)
            noised.append(img_aug)

        for each in copy.deepcopy(np_img_list):
            #   image erosion
            img_aug = aug.add_erode(each)
            eroded.append(img_aug)

        for each in copy.deepcopy(np_img_list):
            #   image dilation
            img_aug = aug.add_dilate(each)
            dilated.append(img_aug)

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', np_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return np_img_list + noised + eroded + dilated

    def split_test_train_write(self, data, idx, font_name,
                               train=70, validate=20, test=10,
                               random_seed=670):
        # create folder names
        train_dir = ROOT_DIR + OUTPUT_DIR + 'train/' + idx
        validate_dir = ROOT_DIR + OUTPUT_DIR + 'validate/' + idx
        test_dir = ROOT_DIR + OUTPUT_DIR + 'test/' + idx

        size = len(data)
        indices = [i for i in range(size)]
        random.seed(random_seed)
        random.shuffle(indices)

        train_count = train / (train+validate+test) * size
        validate_count = validate / (train+validate+test) * size
        split = [[],[],[]]
        count = 0
        while indices:
            i = indices.pop()
            if count < train_count:
                split[0].append(i)
            elif train_count <= count < (train_count+validate_count):
                split[1].append(i)
            else:
                split[2].append(i)
            count += 1


        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        for i in split[0]:
            cv2.imwrite(train_dir + "/" + font_name + str(i) + ".png", data[i])

        if not os.path.exists(validate_dir):
            os.makedirs(validate_dir)
        for i in split[1]:
            cv2.imwrite(validate_dir + "/" + font_name + str(i) + ".png", data[i])

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        for i in split[2]:
            cv2.imwrite(test_dir + "/" + font_name + str(i) + ".png", data[i])

    def gen_img(self, fonts, map_idx2char, max_iter=-1):
        if max_iter<0:
            max_iter = len(map_idx2char)

        iteration = 0
        for idx, char in map_idx2char.items():
            print(idx, char)
            image_list = []
            for font in fonts:
                image_list = self.char2img(idx, char, font)
                self.split_test_train_write(image_list, idx,
                                            font_name=font.split(".")[0])

            iteration += 1
            if iteration >= max_iter:
                break

if __name__ == '__main__':
    # Generating Chinese character
    tick = time.time()

    gen = DataGenerator()
    map_idx_to_char = gen.map_id_char(ROOT_DIR + GB2312_FILE, "mapping_idx_to_char")
    # 微软雅黑 /  黑体 / 仿宋 / 楷体 / 宋体
    fonts = ["msyh.ttc", "simhei.ttf", "simfang.ttf", "simkai.ttf", "simsun.ttc"]
    gen.gen_img(fonts, map_idx_to_char, max_iter=-1)

    print(time.time()-tick)

    # # TESTING
    # font = FONT0
    # idx = '0001'
    # char = map_idx_to_char[idx]
