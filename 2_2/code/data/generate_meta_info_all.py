from os import path as osp
from PIL import Image
import cv2
import numpy as np

from file.utils.misc import scandir


def cal_gradient(img_path):
    img = cv2.imread(img_path)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    return np.mean(sobelxy)


def generate_meta_info_div2k():
    """Generate meta info for dataset.
    """

    gt_folder = 'dataset/train/gt_image_down_all_sub'
    meta_info_txt = 'dataset/train/meta_info_train_all.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img_gd = cal_gradient(osp.join(gt_folder, img_path))
            if img_gd >= 10: 
                img = Image.open(osp.join(gt_folder, img_path))  # lazy load
                width, height = img.size
                mode = img.mode
                if mode == 'RGB':
                    n_channel = 3
                elif mode == 'L':
                    n_channel = 1
                else:
                    raise ValueError(f'Unsupported mode {mode}.')

                info = f'{img_path} ({height},{width},{n_channel}) {img_gd}'
                print(idx + 1, info)
                f.write(f'{info}\n')

def generate_meta_val():
    """Generate meta info for dataset.
    """

    gt_folder = 'dataset/train/gt'
    meta_info_txt = 'dataset/train/meta_info_val_all.txt'

    img_list = sorted(list(scandir(gt_folder)))
    img_list = img_list[int(len(img_list)/10)*9:]

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img_gd = cal_gradient(osp.join(gt_folder, img_path))
            if True: 
                img = Image.open(osp.join(gt_folder, img_path))  # lazy load
                width, height = img.size
                mode = img.mode
                if mode == 'RGB':
                    n_channel = 3
                elif mode == 'L':
                    n_channel = 1
                else:
                    raise ValueError(f'Unsupported mode {mode}.')

                info = f'{img_path} ({height},{width},{n_channel}) {img_gd}'
                print(idx + 1, info)
                f.write(f'{info}\n')

if __name__ == '__main__':
    generate_meta_info_div2k()
    # generate_meta_val()
