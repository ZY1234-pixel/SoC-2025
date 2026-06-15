"""
Official Code Implementation of:
"D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping"
"""

import argparse
import time

import cv2
import glob
import numpy as np
import os
from skimage import measure

from loader.dataset_doc3d_grid_HV import gradient

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from networks.d2dewarp_model import D2DewarpModel

from PIL import Image
from torch import nn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', default=448, help='image size')
    parser.add_argument('--model_path',
                        default='model_weight.pt',
                        help='model path')
    parser.add_argument('--img_path', default='images',
                        help='image path or path to folder containing images (set multi as true)')
    parser.add_argument('--save_path', default='results', help='save path')

    """ Model """
    parser.add_argument('--hv_out_chans', type=int, default=1, help='h line and v line output channels')
    parser.add_argument('--d_model', type=int, default=448, help='last layer dim in UNet and all layers in attention')
    parser.add_argument('--in_chans', type=int, default=4, help='input channels')
    return parser.parse_args()


parser = get_args()

recti_model = D2DewarpModel(img_size=parser.input_size, in_chans=parser.in_chans, hv_out_chans=parser.hv_out_chans,
                            d_model=parser.d_model)

recti_model = torch.nn.DataParallel(recti_model).cuda()
state_dict = torch.load(parser.model_path)
recti_model.load_state_dict(state_dict['state_dict'])
print(f'model loaded')


def predict(img_path, save_path, filename):
    assert os.path.exists(img_path), 'Incorrect Image Path'
    assert os.path.exists(save_path), 'Incorrect Save Path'

    dewarp_path = os.path.join(save_path, 'dewarp/')
    pred_h_path = os.path.join(save_path, 'pred_h/')
    pred_v_path = os.path.join(save_path, 'pred_v/')
    if not os.path.exists(dewarp_path):
        os.makedirs(dewarp_path)
    if not os.path.exists(pred_h_path):
        os.makedirs(pred_h_path)
    if not os.path.exists(pred_v_path):
        os.makedirs(pred_v_path)

    img_size = parser.input_size
    img = np.array(Image.open(img_path).convert("RGB"))

    img_h, img_w, _ = img.shape
    input = cv2.resize(img, (img_size, img_size)) / 255.

    edge = gradient(img)
    edge = edge[:, :, np.newaxis]
    edge = cv2.resize(edge.astype(np.uint8), (img_size, img_size)) / 255.

    # edge1 = cv2.imread("DocHV测试图/1_right_h.png", cv2.IMREAD_GRAYSCALE)
    # edge2 = cv2.imread("DocHV测试图/1_right_v.png", cv2.IMREAD_GRAYSCALE)
    # edge=cv2.bitwise_or(edge1,edge2)
    # edge = cv2.resize(edge, (img_size, img_size))
    # edge = edge.astype(np.float32) / 255.

    recti_model.eval()

    with torch.no_grad():
        input_ = torch.from_numpy(input).permute(2, 0, 1).cuda()
        input_ = input_.unsqueeze(0)
        edge = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).cuda()
        start = time.time()
        pred_h_lst, pred_v_lst, pred_bm = recti_model(torch.cat((input_.float(), edge.float()), dim=1))
        bm = (2. * (pred_bm / (parser.input_size)) - 1) * 1.004
        ps_time = time.time() - start
    # ----------------------------------------------------
    # print("-----bm-----")
    # print(bm)
    bm = bm.cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow

    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2

    out = F.grid_sample(torch.from_numpy(img / 255.).permute(2, 0, 1).unsqueeze(0).float().cuda(), lbl.cuda(),
                        align_corners=True)
    img_geo = ((out[0] * 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
    cv2.imwrite(os.path.join(dewarp_path, filename.rsplit('/', 1)[-1]), img_geo[:, :, ::-1])  # save

    # -----------------------------------------------------
    pred_h = pred_h_lst[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_h = (pred_h > 0.2)
    pred_h = (pred_h * 255.).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_h_path, filename.rsplit('/', 1)[-1]),
                cv2.resize(pred_h, (img_w, img_h)))  # save

    pred_v = pred_v_lst[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_v = (pred_v > 0.2)
    pred_v = (pred_v * 255.).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_v_path, filename.rsplit('/', 1)[-1]),
                cv2.resize(pred_v, (img_w, img_h)))  # save

    return ps_time


if __name__ == '__main__':
    img_path = parser.img_path
    save_path = parser.save_path
    os.makedirs(save_path, exist_ok=True)
    total_time = 0.0

    start = time.time()
    img_num = 0.0

    for file in glob.glob(img_path + "/*"):
        print("file: ", file)
        filename = (save_path + "/" + file[file.rindex("/") + 1:file.rindex(".")] + ".png")

        total_time += predict(file, save_path, filename)
        print("total_time: ", total_time)
        img_num += 1
    print('FPS: %.1f' % (1.0 / (total_time / img_num)))

