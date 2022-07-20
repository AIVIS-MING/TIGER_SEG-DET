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



class CmScorer(object):
    def __init__(self, class_map, incremental=False, ignore_gt_zeros=True,
                 gt_remap={}, pred_remap={}, remap_inplace=False):
        """
        class_map: {label:name}
        incremental: accumulates metrics over repeated calls
        ignore_gt_zeros: ignores 0s in the ground truth
        gt_remap and pred_remap: remap values in target and prediction in-place
        """
        self._ignore_gt_zeros = ignore_gt_zeros
        self._incremental = incremental

        self.gt_remap = gt_remap
        self.pred_remap = pred_remap
        self.remap_inplace=remap_inplace

        self.class_map = class_map

        self.reset()

    def reset(self):
        self.cm = None

    def get_score(self):
        return self._get_score(self.cm)

    def _get_score(self, cm:ConfusionMatrix):
        results = dict(cm_map=cm.table, cm=cm.to_array())
        results['overall_metrics'] = cm.overall_stat
        results['class_metrics'] = cm.class_stat
        results['classes'] = cm.classes
        return results

    def _remap(self, arr, old_new_map):
        #if overlap - have to copy, otherwise could also do inplace
        if old_new_map.values() in old_new_map.keys() or not self.remap_inplace:
            arr_new = arr.copy()
            for old_val, new_val in old_new_map.items():
                arr_new[arr==old_val] = new_val
            arr = arr_new
        else:
            for old_val, new_val in old_new_map.items():
                arr[arr==old_val] = new_val
        return arr

    def __call__(self, gt, pred, show=False):
        """
        gt: ground truth numpy array
        pred: pred numpy array
        returns the confusion matrix and metrics for the given ground-truth and pred mask arrays as dict {label:dice}.
        call get_score in the incremental case to get the full score
        """
        if not self._incremental:
            self.reset()
        gt = self._remap(gt, self.gt_remap)
        pred = self._remap(pred, self.pred_remap)
        if self._ignore_gt_zeros:
            pred = pred[gt!=0]
            gt = gt[gt!=0]

        class_labels = list(sorted(self.class_map.keys()))
        class_names = [self.class_map[k] for k in class_labels]
        cm_arr = confusion_matrix(gt, pred, labels=class_labels)

        matrix = {}
        for i, cl_true in enumerate(class_names):
            cl_matrix = {}
            matrix[cl_true] = cl_matrix
            for j,cl_pred in enumerate(class_names):
                cl_matrix[cl_pred] = int(cm_arr[i,j])

        cm = ConfusionMatrix(matrix=matrix)

        if self.cm is None:
            self.cm = cm
        else:
            self.cm = self.cm.combine(cm)

        if show:
            self.cm.plot(normalized=True)
            plt.show()

        return self._get_score(cm)

class TigerSegmScorer(CmScorer):
    def __init__(self, incremental=True, **kwargs):
        #1: Invasive Tumor, 2: Tumor-assoc. Stroma, 3: DCIS, 4: Healthy, 5: Necrosis, 6: Inflamed Stroma, 7: Rest
        gt_remap={4:3, 5:3, 6:2, 7:3}

        #pred mask: map all other-classes to 3
        pred_remap = {k:3 for k in range(256)}
        pred_remap.update({1:1, 2:2, 6:2})
        super().__init__(class_map={1:'Tumor',2:'Stroma',3:'Rest'}, incremental=incremental,
                         gt_remap=gt_remap, pred_remap=pred_remap, ignore_gt_zeros=True, **kwargs)

    def _get_score(self, *args, **kwargs):
        """ returns just the cm and the dice metrics """
        metrics = super()._get_score(*args, **kwargs)
        dice_metrics = metrics['class_metrics']['F1']
        dice_metrics['cm'] = metrics['cm']
        dice_metrics['classes'] = metrics['classes']
        return dice_metrics

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
                cv2.imshow("img_ori", img)

            cv2.waitKey(0)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
            cv2.waitKey(2)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def slide_inference(img, model,test_pipeline):
    test_pipeline.transforms[1].img_scale = [(512,512)]
    h_stride, w_stride = 256, 256
    h_crop, w_crop = 512, 512
    h_img, w_img = img.shape[0], img.shape[1]
    num_classes = NUM_CLASS
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    preds = torch.tensor((), dtype=torch.float64)
    preds = preds.new_zeros((1, num_classes, h_img, w_img)).cuda()

    # results = np.zeros((h_img, w_img), dtype=np.uint8)

    count_mat = torch.tensor((), dtype=torch.float64)
    count_mat = count_mat.new_zeros((1,1, h_img, w_img)).cuda()

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[y1:y2, x1:x2, :]
            # cv2.imshow("", crop_img)
            # cv2.waitKey(0)
            data = dict()
            data = dict(img=crop_img)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, ['cuda:0'])[0]
            else:
                data['img_metas'] = [i.data[0] for i in data['img_metas']] 

            model.eval()
            with torch.no_grad():
                crop_seg_logit, seg_logit = model(return_loss=False, rescale=False, **data)
            preds += F.pad(crop_seg_logit,
                            (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    preds = preds / count_mat

    seg_logit = F.softmax(preds, dim=1)
    seg_pred = seg_logit.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()
    return seg_pred[0]

def whole_inference(img, model, test_pipeline):
    img = cv2.resize(img, (512,512))
    data = dict()
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, ['cuda:0'])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']] 

    model.eval()
    with torch.no_grad():
        crop_seg_logit, seg_logit = model(return_loss=False, rescale=False, **data)
    SegResult = cv2.resize(seg_logit[0], (2048,2048), interpolation=cv2.INTER_NEAREST)
    return SegResult


if __name__=='__main__':
    # In function init_segmentor, the model is set to evaluatio mode.
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    config = mmcv.Config.fromfile(config_file)
    test_pipeline = [LoadImage()] + config.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)


    txt = open(VAL_TXT_PATH, 'r')
    lines = txt.readlines()
    length = len(lines)

    gt_l = []
    seg_l = []
    scorer = TigerSegmScorer(incremental=True)
    for i in range(0, length, 1):
        line = lines[i]
        # print("Processing=============:{}-- {}/{}".format(line, i, length))
        img_name = line.split('\n')[0]

        gt = cv2.imread(osp.join(MASK_PATH, img_name))
        gt = gt[:,:, 0]
        ROI_mask = gt.copy()
        ROI_mask[ROI_mask!=0] = 1

        img = cv2.imread(osp.join(IMAGE_PATH, img_name))
        SegResult = inference_segmentor(model, img)[1][0]

        # seg_l.append(SegResult * ROI_mask)
        # gt_l.append(gt)
        # f_score = mmseg.core.evaluation.mean_fscore(seg_l, gt_l, 3, ignore_index= 255, reduce_zero_label=True)


        #result = result + 1
        # show_preded_result(img, gt, palette=PALETTE_GT, win_name='GT', show=True, wait_time=1, out_file=None, opacity=0.3)
        # show_preded_result(img, SegResult+1, palette=PALETTE_GT, win_name='pred', show=True, wait_time=0, out_file=None, opacity=0.3)

        # SegResult = slide_inference(img, model, test_pipeline)
        SegResult = SegResult + 1
        seg_flat = SegResult[ROI_mask == 1]
        gt_flat = gt[ROI_mask == 1]
        score = scorer(gt_flat, seg_flat)
    result = scorer.get_score()
    print("The real-time avg is: Rest: {}  Stroma: {}   Tumor:  {}".format(result['Rest'], result['Stroma'], result['Tumor']))