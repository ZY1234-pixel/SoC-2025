import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from file.utils.misc import scandir

def main():
    """图像裁剪
    opt：配置字典，包含：
    n_thread (int)：线程数。
    compression_level (int)：CV_IMWRITE_PNG_COMPRESSION，取值范围0至9。
        数值越高表示文件越小，压缩时间越长。
        使用0可实现更快的CPU解压缩。默认值：3，与cv2相同。
    input_folder (str)：输入文件夹路径。
    save_folder (str)：保存文件夹路径。
    crop_size (int)：裁剪尺寸。
    step (int)：滑动窗口的步长。
    thresh_size (int)：阈值尺寸。小于该尺寸的图像块将被舍弃。
使用说明：
    对每个文件夹运行此脚本。
    数据集通常包含四个待处理文件夹：
        train_HR
        train_LR_bicubic/X2
        train_LR_bicubic/X3
        train_LR_bicubic/X4
    处理后，每个子文件夹应包含相同数量的子图像。
    请根据自身设置修改opt配置文件。
    """

    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3

    # HR images
    opt['input_folder'] = 'dataset/train/gt'
    opt['save_folder'] = 'dataset/train/gt_image_down_sub'
    opt['crop_size'] = 1024
    opt['step'] = 512
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # input images 
    opt['input_folder'] = 'dataset/train/input'
    opt['save_folder'] = 'dataset/train/blur_image_down_sub'
    opt['crop_size'] = 1024
    opt['step'] = 512
    opt['thresh_size'] = 0
    extract_subimages(opt)

def extract_subimages(opt):
    """将图像裁剪为子图像。
    参数：
        opt (dict)：配置字典。包含：
            input_folder (str)：输入文件夹路径。
            save_folder (str)：保存文件夹路径。
            n_thread (int)：线程数。
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))
    length = len(img_list)
    img_list = img_list[:int(length/10)*9]

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """对每个进程
    参数：
        path (str)：图像路径。
        opt (dict)：配置字典。包含：
            crop_size (int)：裁剪尺寸。
            step (int)：重叠滑动窗口的步长。
            thresh_size (int)：阈值尺寸。尺寸小于 thresh_size 的补丁将被舍弃。
            save_folder (str)：保存文件夹路径。
            compression_level (int)：用于 cv2.IMWRITE_PNG_COMPRESSION 的压缩级别。
    返回值：
        process_info (str)：进度条中显示的进程信息。
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()
