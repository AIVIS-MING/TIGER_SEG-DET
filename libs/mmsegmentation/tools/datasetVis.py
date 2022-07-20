import os.path as osp
import os
from re import T
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import cv2
import mmcv
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
from mmseg.ops import resize
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from pycm import ConfusionMatrix
from matplotlib import pyplot as plt
import mmseg
import pandas as pd
from tqdm import tqdm
import warnings


DATA_DIR_SAVE = "/media/mingfan/DATASSD/TIGER/mmsegmentation/WEIGHT/Result"
DATA_DIR = "/media/mingfan/DataHDD/DATA_Tiger/Seg_new4"
IMAGE_PATH = osp.join(DATA_DIR, "images") #images_patch
MASK_PATH = osp.join(DATA_DIR, "masks") #images_patch

VAL_TXT_PATH = osp.join(DATA_DIR, "val_1_fold_512_4x.txt")
WEIGHT_DIR = "/media/mingfan/DATASSD/TIGER/mmsegmentation/WEIGHT/uppernet-Knet_512_r50_bs4_seg4"

config_file = osp.join(WEIGHT_DIR, 'kent_tiger.py')
checkpoint_file = osp.join(WEIGHT_DIR, 'iter_6000.pth')

CLASSES_GT = ('bg', 'invasive tumor', 'tumor-asso stoma', 'in-situ tumor', 'healthy glands', 
            'necrosis not in-situ', 'inflamed stroma', 'rest')
PALETTE_GT = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], 
            [255, 255, 0], [0,255,255], [255, 0, 255]]


CLASSES_PRED = ('invasive tumor', 'tumor-asso stoma', 'in-situ tumor', 'healthy glands', 
            'necrosis not in-situ', 'inflamed stroma', 'rest')
PALETTE_PRED = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], 
            [255, 255, 0], [0,255,255], [255, 0, 255]]

CLASSES_PRED_3 = ('invasive tumor', 'tumor-asso stoma','rest')
PALETTE_PRED_3 = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]

NUM_CLASS = 7

def show_preded_result(img,
                seg,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,   
                out_file=None,
                opacity=0.3):

        palette = np.array(palette)
        assert palette.shape[0] == len(palette)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label] = color

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        img_ori = img.copy()
        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img) 
            
            if opacity>=0.6:
                cv2.namedWindow("img_ori", cv2.WINDOW_NORMAL)
                cv2.imshow("img_ori", img_ori)

            cv2.waitKey(0)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
            cv2.waitKey(2)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

if __name__=='__main__':

    txt = open(VAL_TXT_PATH, 'r')
    lines = txt.readlines()
    length = len(lines)

    gt_l = []
    seg_l = []
    for i in range(0, length, 1):
        line = lines[i]
        # print("Processing=============:{}-- {}/{}".format(line, i, length))
        img_name = line.split('\n')[0]

        gt = cv2.imread(osp.join(MASK_PATH, img_name))
        gt = gt[:,:, 0]
        ROI_mask = gt.copy()
        ROI_mask[ROI_mask!=0] = 1
        img = cv2.imread(osp.join(IMAGE_PATH, img_name))
        show_preded_result(img, gt, palette=PALETTE_GT, win_name='GT', show=True, wait_time=1, out_file=None, opacity=0.7)
