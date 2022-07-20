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
import warnings

DATA_DIR_SAVE = "/workspace/DATASET/TIGER-Dataset/RESULT_TRAIN"


DATA_DIR = "/workspace/DATASET/TIGER-Dataset/wsirois/roi-level-annotations/tissue-bcss"
IMAGE_PATH = osp.join(DATA_DIR, "images") #images_patch
VAL_TXT_PATH = "/workspace/DATASET/TIGER-Dataset/wsirois/roi-level-annotations/tissue-bcss/TXT_File/train_1_fold_512_4x.txt"
TRAIN_TXT_PATH = osp.join(DATA_DIR, "train_1_fold_1030.txt")
WEIGHT_DIR = "/workspace/DATASET/TIGER-Dataset/wsirois/SEGWEIGHT/fold_1_uppernet-Knet_512_4x-SGD_0.0001"

config_file = osp.join(WEIGHT_DIR, 'kent_tiger.py')
checkpoint_file = osp.join(WEIGHT_DIR, 'iter_20000.pth')

CLASSES = ('background','invasive tumor', 'tumor-asso stoma', 'in-situ tumor', 'healthy glands', 
            'necrosis not in-situ', 'inflamed stroma', 'rest')
PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], 
            [255, 255, 0], [0,255,255], [255, 0, 255]]


def show_result(img_path,
                label_path,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,   
                out_file=None,
                opacity=0.3):
        img_ori = mmcv.imread(img_path)
        #img_ori = cv2.resize(img_ori, dsize=(1024, 1024), interpolation=cv2.INTER_AREA)
        img = img_ori.copy()

        seg = mmcv.imread(label_path)
        #seg = cv2.resize(seg, dsize=(1024, 1024), interpolation=cv2.INTER_AREA)
        seg = seg[:,:,0]
        # seg = seg.copy()

        palette = np.array(PALETTE)
        assert palette.shape[0] == len(CLASSES)
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
                cv2.imshow("img_ori", img_ori_)

            cv2.waitKey(0)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
            cv2.waitKey(2)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img

def show_preded_result(img,
                seg,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,   
                out_file=None,
                opacity=0.3):

        palette = np.array(PALETTE)
        assert palette.shape[0] == len(CLASSES)
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


def slide_inference(img, model, rescale, ori_size, test_pipeline):
    h_stride, w_stride = 256, 256
    h_crop, w_crop = 512, 512
    h_img, w_img = img.shape[0], img.shape[1]
    num_classes = 8
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
    if rescale:
        preds = resize(
            preds,
            size=ori_size,
            mode='bilinear',
            align_corners=False,
            warning=False)

    seg_logit = F.softmax(preds, dim=1)
    seg_pred = seg_logit.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()
    return seg_pred

# In function init_segmentor, the model is set to evaluatio mode.
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
config = mmcv.Config.fromfile(config_file)
test_pipeline = [LoadImage()] + config.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)


txt = open(VAL_TXT_PATH, 'r')
lines = txt.readlines()
length = len(lines)
for i in range(0, length, 1):
    line = lines[i]
    print("Processing=============:{}-- {}/{}".format(line, i, length))
    img_path = line.split('\n')[0]
    img_name = img_path.split('/')[-1].split('.jpg')[0]
    mask_path = img_path.replace('images_patch_512_4x', 'masks_patch_512_4x')
    mask_path = mask_path.replace('jpg', 'png')
    mask_name = "{}/{}_mask.jpg".format(DATA_DIR_SAVE, img_name)
    overlay_img_mask = show_result(img_path=img_path, label_path=mask_path, palette=PALETTE, 
                                   show=False, win_name="overlay", opacity=0.6,
                                   out_file=None)
      
    img = cv2.imread(img_path)
    # img = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_AREA)

    img_size = img.shape[0:2]
    SegResult = slide_inference(img, model, True, img_size, test_pipeline)

    pred_name = "{}/{}_pred.jpg".format(DATA_DIR_SAVE, img_name)
    overlay_img_pred = show_preded_result(img=img, seg=SegResult[0], palette=PALETTE, 
                                          show=False, opacity=0.6, win_name="preded-result",
                                          out_file=None)
    Result = cv2.hconcat([img, overlay_img_mask, overlay_img_pred])
    cv2.imwrite(pred_name, Result)
    
    

# for file in os.listdir(IMAGE_PATH):
#     if(file.split('.')[-1] != "png"):
#         continue
    
#     img_path = osp.join(IMAGE_PATH,file)
#     mask_path = img_path.replace('images', 'masks')
#     overlay_img_mask = show_result(img_path=img_path, label_path=mask_path, palette=PALETTE, show=False, win_name="overlay", opacity=0.6)
      
#     img = cv2.imread(img_path)
#     img_size = img.shape[0:2]
#     SegResult = slide_inference(img, model, True, img_size, test_pipeline)
#     overlay_img_pred = show_preded_result(img=img, seg=SegResult[0], palette=PALETTE, show=False, opacity=0.5, win_name="preded-result")

