import os
import copy
import pickle
import argparse
import cv2

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Set ROOT_DIR to "data_synthesis/" when run in PyCharm's Console, Set to "" otherwise
ROOT_DIR = "data_synthesis/"
FONT_DIR = 'fonts/'
OUTPUT_DIR = 'data/'



GB2312_FILE = "GB2312-UTF8_2.txt"




HEIGHT = 100
WIDTH = 100
MARGIN = 4

FONT0 = 'msyh.ttc'

# Find minimum bounding box
class FindImageBBox(object):
    def __init__(self, ):
        pass

    def do(self, img):
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

def map_id_char(char_path, map_backup = ""):
    # read all chars in GB2312 txt file into memory
    all_char = ""
    with open(file=char_path, mode='r', encoding='utf-8') as f:
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
    mapping_char_to_idx = {}
    for idx in range(len(all_char)):
        key = str(idx+1).zfill(4)
        val = all_char[idx]#.encode('utf-8')
        mapping_idx_to_char[key] = val
        #mapping_char_to_idx[val] = key
    if map_backup:
        with open(map_backup + ".pkl", 'wb') as f:
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


map_idx_to_char = map_id_char(ROOT_DIR + GB2312_FILE, "mapping_idx_to_char")

def font_to_img():
    pass


"""
def gen_img(font, map_idx2char):
    for char, value in map_idx2char.items():  # 外层循环是字
        image_list = []
        print(char, value)
        # char_dir = os.path.join(images_dir, "%0.5d" % value)
        for j, verified_font_path in enumerate(verified_font_paths):  # 内层循环是字体
            if rotate == 0:
                image = font2image.do(verified_font_path, char)
                image_list.append(image)
            else:
                for k in all_rotate_angles:
                    image = font2image.do(verified_font_path, char, rotate=k)
                    image_list.append(image)
"""

def char2img(idx, char, font, output_path):
    # create folder
    train_dir = ROOT_DIR + OUTPUT_DIR + 'train/' + idx
    validate_dir = ROOT_DIR + OUTPUT_DIR + 'validate/' + idx
    test_dir = ROOT_DIR + OUTPUT_DIR + 'test/' + idx

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(validate_dir):
        os.makedirs(validate_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

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

    # retrieve 3-channel matrix of image
    img_data = list(img.getdata())
    sum_val = 0
    for i in img_data:
        sum_val += sum(i)
    if sum_val > 2:
        find_image_bbox = FindImageBBox()
        np_img = np.asarray(img_data, dtype='uint8')
        np_img = np_img[:, 0]
        np_img = np_img.reshape((HEIGHT, WIDTH))
        cropped_box = find_image_bbox.do(np_img)
        left, upper, right, lower = cropped_box
        np_img = np_img[upper: lower + 1, left: right + 1]

    #   gen. image pepper-salt noise

    #   image erosion
    #   image dilation

    return image_list