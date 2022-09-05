from time import time
from functools import wraps

from pathlib import Path
from pickle import NONE
from typing import List


import numpy as np
from tqdm import tqdm
from mmseg.apis import init_segmentor, inference_segmentor
import mmcv
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
from mmseg.ops import resize
import torch.nn.functional as F
import warnings
import cv2
import os.path as osp
import math

#For Unet
import torch.nn as nn
from torchvision import transforms as T

import sys
sys.path.append("/home/user/yolov5")
from models.experimental import attempt_load
from models.common import DetectMultiBackend
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

from mmdet.apis import init_detector, inference_detector

from ensemble_boxes import *

from .gcio import (
    TMP_DETECTION_OUTPUT_PATH,
    TMP_SEGMENTATION_OUTPUT_PATH,
    TMP_TILS_SCORE_PATH,
    copy_data_to_output_folders,
    get_image_path_from_input_folder,
    get_tissue_mask_path_from_input_folder,
    initialize_output_folders,
)
from .rw import (
    READING_LEVEL,
    WRITING_TILE_SIZE,
    DetectionWriter,
    SegmentationWriter,
    TilsScoreWriter,
    open_multiresolutionimage_image,
)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap

def detect_mm(img_batch, model, size_patch, bs):    
    pred = []
    for i in range (0, len(img_batch), bs) :
        img_batch_inf = img_batch[i:min(i + bs, len(img_batch))]
        if len(img_batch_inf) % bs != 0 :
            while(len(img_batch_inf) % bs != 0) :
                img_batch_inf.append(np.zeros((size_patch, size_patch,3)))
            
        pred = pred + inference_detector(model, img_batch_inf)
    return pred

def detect_yolo(img0, model, imgsz_model, bs, device):
    #AUGMENT = False
    conf_thres = 0.2
    iou_thres = 0.35
    
    detectionClass = [0]
    #AGNOSTIC_NMS = False
    stride, names, pt = model.stride, model.names, model.pt
        
    imgsz_model = [imgsz_model, imgsz_model]
    imgsz_model = check_img_size(imgsz_model, s=stride)  # check image size
    
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz_model))  # warmup
    
    # Convert
    det_batch = []
    for i in range (0, len(img0), bs):
        img_batch = []
        img0_shape_batch = []
        for j in range (i, min(i+bs, len(img0))):
            img = letterbox(img0[j], imgsz_model, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
                
            # Padded resize
            img = torch.from_numpy(img).to(device)
            img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # if len(img.shape) == 3:
            #     img = img[None]  # expand for batch dim
            img_batch.append(img)
            img0_shape_batch.append(img0[0].shape[0:2])
            
        # Inference
        img_batch = torch.stack(img_batch)
        out = model(img_batch, augment=False)
        # Apply NMS
        out = non_max_suppression(out, conf_thres, iou_thres, classes=detectionClass, max_det = 300, agnostic = True)
    
        for si, pred in enumerate(out):
            if len(pred) == 0:
                det_batch.append([])
                continue
            predn = pred.clone()
            scale_coords(img_batch[si].shape[1:], predn[:, :4], [img0_shape_batch[si][0], img0_shape_batch[si][1]]).round()  # native-space pred
            det_batch.append(predn)
            
    return det_batch
    

def slide_inference(img, model_1, model_2, model_swin, test_pipeline):
    h_stride, w_stride = 512, 512
    h_crop, w_crop = 1024, 1024
    h_img, w_img = img.shape[0], img.shape[1]
    num_classes = 3
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    preds_1 = torch.tensor((), dtype=torch.float64)
    preds_1 = preds_1.new_zeros((1, num_classes, h_img, w_img)).cuda()

    preds_2 = torch.tensor((), dtype=torch.float64)
    preds_2 = preds_2.new_zeros((1, num_classes, h_img, w_img)).cuda()

    # preds_swin = torch.tensor((), dtype=torch.float64)
    # preds_swin = preds_swin.new_zeros((1, num_classes, h_img, w_img)).cuda()

    count_mat = torch.tensor((), dtype=torch.float64)
    count_mat = count_mat.new_zeros((1,1, h_img, w_img)).cuda()
    model_1.eval()
    model_2.eval()
    #model_swin.eval()

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
            if next(model_1.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, ['cuda:0'])[0]
            else:
                data['img_metas'] = [i.data[0] for i in data['img_metas']] 
            
            with torch.no_grad():
                # torch.cuda.empty_cache()
                crop_seg_logit_1, seg_logit_1 = model_1(return_loss=False, rescale=False, **data)
                # torch.cuda.empty_cache()
                crop_seg_logit_2, seg_logit_2 = model_2(return_loss=False, rescale=False, **data)
                # torch.cuda.empty_cache()
                # crop_seg_logit_swin, seg_logit_swin = model_swin(return_loss=False, rescale=False, **data)
            preds_1 += F.pad(crop_seg_logit_1,(int(x1), int(preds_1.shape[3] - x2), int(y1), int(preds_1.shape[2] - y2)))
            preds_2 += F.pad(crop_seg_logit_2,(int(x1), int(preds_2.shape[3] - x2), int(y1),int(preds_2.shape[2] - y2)))
            # preds_swin += F.pad(crop_seg_logit_swin,(int(x1), int(preds_swin.shape[3] - x2), int(y1),int(preds_swin.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    
    assert (count_mat == 0).sum() == 0
    preds_1 = preds_1 / count_mat
    preds_2 = preds_2 / count_mat
    # preds_swin = preds_swin / count_mat

    seg_logit_1 = F.softmax(preds_1, dim=1)
    seg_logit_2 = F.softmax(preds_2, dim=1)
    # seg_logit_swin = F.softmax(preds_swin, dim=1)

    # seg_logit_avg = (seg_logit_1 + seg_logit_2 + seg_logit_swin) / 3
    seg_logit_avg = (seg_logit_1 + seg_logit_2 ) / 2
    seg_pred = seg_logit_avg.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()
    
    return seg_pred[0]


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
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


def process_image_tile_to_bulk_segmentation(
    image_tile: np.ndarray, 
    model,
    device,
    compress: int
)-> np.ndarray:

    toTensor = T.ToTensor()
    Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image_tile = cv2.cvtColor(image_tile.astype(np.uint8), cv2.COLOR_BGR2RGB)
    t_img = Norm_(toTensor(image_tile))
    with torch.no_grad():
        model.eval()
        t_img = t_img.to(device)
        b_img = torch.unsqueeze(t_img, 0)
        pred = model(b_img)
        logit = torch.argmax(pred, dim=1).cpu().numpy()[0]  
    
    # resize logit from compression factor
    logit = cv2.resize(logit, dsize=(int(logit.shape[1]/compress), int(logit.shape[0]/compress)), interpolation=cv2.INTER_NEAREST)
    return logit

@timing
def process_image_tile_to_segmentation(image_tile_seg: np.ndarray, tissue_mask_tile: np.ndarray, 
                                       model_res101_folder1, 
                                       model_res101_folder2, 
                                       model_swin, 
                                       test_pipeline,
                                       Padding) -> np.ndarray:
    """
    class_3: 1 2 3 --> 0 1 2
    class_4: 1 2 3 4 --> 0 1 2 3
    """
    image_tile_seg = cv2.cvtColor(image_tile_seg, cv2.COLOR_BGR2RGB)
    prediction_tumor = slide_inference(img=image_tile_seg, 
                                       model_1=model_res101_folder1, 
                                       model_2=model_res101_folder2, 
                                       model_swin=model_swin, 
                                       test_pipeline=test_pipeline)

    prediction_fusion = prediction_tumor + 1
    tissue_mask_tile[tissue_mask_tile!=0] = 1
    prediction_final_rm_padding = prediction_fusion[Padding: (Padding+tissue_mask_tile.shape[0]), Padding:(Padding+tissue_mask_tile.shape[1])]
     
    torch.cuda.empty_cache()  
    return prediction_final_rm_padding * tissue_mask_tile

@timing
def process_image_tile_to_detections(
    image_tile: np.ndarray,
    tissue_mask: np.ndarray,
    model1,
    model2,
    model5,
    model6,
    model7,
    device
) -> List[tuple]:

    image_tile = cv2.cvtColor(image_tile, cv2.COLOR_BGR2RGB)

    if not np.any(tissue_mask == 1):
        return []
    
    xs = []
    ys = []
    probabilities = []    

    boxes = [[], [], [], [], []]
    scores = [[], [], [], [], []]
    labels = []

    ##################256#####################
    size_patch = 256
    #interval_patch = 224
    interval_patch = 232
    diff_size = int((size_patch - interval_patch)/2)
    
    img_batch = []
    coor_tiles = []
        
    for iH in range (0, image_tile.shape[0], interval_patch):
        for iW in range (0, image_tile.shape[1], interval_patch):
            begin_x = iW
            end_x = iW + size_patch - 1
            begin_y = iH
            end_y = iH + size_patch - 1
            if end_y >= image_tile.shape[0]:
                end_y = image_tile.shape[0] - 1
            if end_x >= image_tile.shape[1]:
                end_x = image_tile.shape[1] - 1
                       
            cropTissue = tissue_mask[begin_y:end_y, begin_x:end_x]            
            if not np.any(cropTissue == 1):
                continue
            
            cropImg = np.zeros((size_patch, size_patch,3)) + 128
            cropImg[0:end_y-begin_y+1, 0:end_x-begin_x+1, :] = image_tile[begin_y:end_y+1, begin_x:end_x+1, :]

            img_batch.append(cropImg)
            coor_tiles.append([begin_x, begin_y, end_x, end_y])
            
    
    tissue_mask[tissue_mask != 0] = 1

    kernel = np.ones((3,3), np.uint8) 
    if tissue_mask.sum() < image_tile.shape[0] * image_tile.shape[1] * 0.3:
        tissue_mask = cv2.erode(tissue_mask, kernel, iterations = 10)

    torch.cuda.empty_cache()
    pred = detect_yolo(img_batch, model1, 256, _bs_pred1_det, select_device(device))
    torch.cuda.empty_cache()
    pred2 = detect_mm(img_batch, model=model2, size_patch = size_patch, bs = _bs_pred2_det)
    
    pred_list = []
    pred_list2 = []
    for i in range(0, len(coor_tiles)):
        if len(pred[i]) == 0 :
            pred_list.append([])
        else :
            pred_list.append(pred[i].tolist())
        pred_list2.append(pred2[i][0].tolist())
    
        boxes_list = []
        scores_list = []
        
        if len(pred_list[i]) != 0 :
            tmp = np.delete(pred_list[i], 4, 1)
            tmp = np.delete(tmp, 4, 1)
            boxes_list.append(tmp.tolist())
            score = list(zip(*pred_list[i]))[4]
            scores_list.append(score)  
        
        if len(pred_list2[i]) != 0 :
            boxes_list.append((np.delete(pred_list2[i], 4, 1)).tolist())
            score = list(zip(*pred_list2[i]))[4]
            scores_list.append(score)       
                        
        for k in range(0, len(boxes_list)):
            for k2 in range(0, len(boxes_list[k])):            
                lt_x = coor_tiles[i][0] + boxes_list[k][k2][0]
                lt_y = coor_tiles[i][1] + boxes_list[k][k2][1]
                rb_x = coor_tiles[i][0] + boxes_list[k][k2][2]
                rb_y = coor_tiles[i][1] + boxes_list[k][k2][3]
                cx = (lt_x + rb_x)/2
                cy = (lt_y + rb_y)/2
                
                if cy > image_tile.shape[0] or cx > image_tile.shape[1]:
                    continue
                
                if tissue_mask[int(cy)][int(cx)] != 1:
                    continue            
                
                if (cx > coor_tiles[i][0] + size_patch - diff_size) or \
                    (cy > coor_tiles[i][1] + size_patch - diff_size) or \
                    (cx < coor_tiles[i][0] + diff_size) or (cy < coor_tiles[i][1] + diff_size):
                    if (tissue_mask[int(lt_y)][int(lt_x)] != 0 and \
                        tissue_mask[min(int(rb_y), tissue_mask.shape[0]-1)][min(int(rb_x), tissue_mask.shape[1]-1)] != 0):
                        continue            
            
                boxes[k].append([coor_tiles[i][0] + boxes_list[k][k2][0], coor_tiles[i][1] + boxes_list[k][k2][1], coor_tiles[i][0] + boxes_list[k][k2][2], coor_tiles[i][1] + boxes_list[k][k2][3]])
                scores[k].append(scores_list[k][k2])
                
    ##################384#####################
    size_patch = 384
    #interval_patch = 224
    interval_patch = 360
    diff_size = int((size_patch - interval_patch)/2)
    
    img_batch = []
    coor_tiles = []
        
    for iH in range (0, image_tile.shape[0], interval_patch):
        for iW in range (0, image_tile.shape[1], interval_patch):
            begin_x = iW
            end_x = iW + size_patch - 1
            begin_y = iH
            end_y = iH + size_patch - 1
            if end_y >= image_tile.shape[0]:
                end_y = image_tile.shape[0] - 1
            if end_x >= image_tile.shape[1]:
                end_x = image_tile.shape[1] - 1
                       
            cropTissue = tissue_mask[begin_y:end_y, begin_x:end_x]            
            if not np.any(cropTissue == 1):
                continue
            
            cropImg = np.zeros((size_patch, size_patch,3)) + 128
            cropImg[0:end_y-begin_y+1, 0:end_x-begin_x+1, :] = image_tile[begin_y:end_y+1, begin_x:end_x+1, :]

            img_batch.append(cropImg)
            coor_tiles.append([begin_x, begin_y, end_x, end_y])
            
    torch.cuda.empty_cache()
    pred5 = detect_yolo(img_batch, model5, 1280, _bs_pred5_det, select_device(device))
    torch.cuda.empty_cache()
    pred6 = detect_mm(img_batch, model=model6, size_patch = size_patch, bs = _bs_pred6_det)
    torch.cuda.empty_cache()
    pred7 = detect_mm(img_batch, model=model7, size_patch = size_patch, bs = _bs_pred7_det)
    
    pred_list5 = []
    pred_list6 = []
    pred_list7 = []
    for i in range(0, len(coor_tiles)):
        if len(pred5[i]) == 0 :
            pred_list5.append([])
        else :
            pred_list5.append(pred5[i].tolist())
        pred_list6.append(pred6[i][0].tolist())
        pred_list7.append(pred7[i][0].tolist())
    
        boxes_list = []
        scores_list = []
        if len(pred_list5[i]) != 0 :
            tmp = np.delete(pred_list5[i], 4, 1)
            tmp = np.delete(tmp, 4, 1)
            boxes_list.append(tmp.tolist())
            score = list(zip(*pred_list5[i]))[4]
            scores_list.append(score)        
        if len(pred_list6[i]) != 0 :
            boxes_list.append((np.delete(pred_list6[i], 4, 1)).tolist())
            score = list(zip(*pred_list6[i]))[4]
            scores_list.append(score)    
        if len(pred_list7[i]) != 0 :
            boxes_list.append((np.delete(pred_list7[i], 4, 1)).tolist())
            score = list(zip(*pred_list7[i]))[4]
            scores_list.append(score)     
        
        for k in range(0, len(boxes_list)):
            for k2 in range(0, len(boxes_list[k])):       
                lt_x = coor_tiles[i][0] + boxes_list[k][k2][0]
                lt_y = coor_tiles[i][1] + boxes_list[k][k2][1]
                rb_x = coor_tiles[i][0] + boxes_list[k][k2][2]
                rb_y = coor_tiles[i][1] + boxes_list[k][k2][3]
                cx = (lt_x + rb_x)/2
                cy = (lt_y + rb_y)/2
                
                if cy > image_tile.shape[0] or cx > image_tile.shape[1]:
                    continue
                
                if tissue_mask[int(cy)][int(cx)] != 1:
                    continue            
                
                if (cx > coor_tiles[i][0] + size_patch - diff_size) or \
                    (cy > coor_tiles[i][1] + size_patch - diff_size) or \
                    (cx < coor_tiles[i][0] + diff_size) or (cy < coor_tiles[i][1] + diff_size):
                    if (tissue_mask[int(lt_y)][int(lt_x)] != 0 and \
                        tissue_mask[min(int(rb_y), tissue_mask.shape[0]-1)][min(int(rb_x), tissue_mask.shape[1]-1)] != 0):
                        continue            
            
                boxes[2 + k].append([coor_tiles[i][0] + boxes_list[k][k2][0], coor_tiles[i][1] + boxes_list[k][k2][1], coor_tiles[i][0] + boxes_list[k][k2][2], coor_tiles[i][1] + boxes_list[k][k2][3]])
                scores[2 + k].append(scores_list[k][k2])
        
    weights_all = []
    for i in range(0, len(boxes)): 
        labels.append(np.ones(len(boxes[i])).tolist())
        boxes[i] = np.array(boxes[i])      
        if len(boxes[i]) != 0:   
            boxes[i][:,0] /= image_tile.shape[1]
            boxes[i][:,2] /= image_tile.shape[1]
            boxes[i][:,1] /= image_tile.shape[0]
            boxes[i][:,3] /= image_tile.shape[0]
        weights_all.append(1)
        
    iou_thr = 0.4
    skip_box_thr = 0.1
    
    boxes_fusion, scores_fusion, labels_fusion = weighted_boxes_fusion(boxes, scores, labels, weights=weights_all, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
              
    boxes_fusion[:,0] *= image_tile.shape[1]
    boxes_fusion[:,2] *= image_tile.shape[1]
    boxes_fusion[:,1] *= image_tile.shape[0]
    boxes_fusion[:,3] *= image_tile.shape[0]

    for i in range(0, len(boxes_fusion)):
        xs.append(float((boxes_fusion[i][0] + boxes_fusion[i][2])/2))
        ys.append(float((boxes_fusion[i][1] + boxes_fusion[i][3])/2))
        probabilities.append(float(scores_fusion[i]))
        
    return list(zip(xs, ys, probabilities))

def process_segmentation_detection_to_tils_score(
    segmentation_path: Path, detections: List[tuple]
) -> int:
    """Example function that shows processing a segmentation mask and corresponding detection for the computation of a tls score.
    
    NOTE 
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        segmentation_mask (np.ndarray): [description]
        detections (List[tuple]): [description]

    Returns:
        int: til score (between 0, 100)
    """
    # image = open_multiresolutionimage_image(path=segmentation_path)
    # width, height = image.getDimensions()
    
    # level = 0
    # tile_size = WRITING_TILE_SIZE # should be a power of 2
    # stroma_area = 0
    # for y in range(0, height, tile_size):
    #     for x in range(0, width, tile_size):
    #         seg_result = image.getUCharPatch(startX=x, startY=y, width=tile_size, height=tile_size, level=level).squeeze()
    #         stroma_currentArea = len(np.where(seg_result == 2)[0])
    #         stroma_area += stroma_currentArea
            
    # if len(detections) == 0:
    #     print("detection_len = 0")
    #     return 1
    # if stroma_area == 0:
    #     print("stroma_area = 0")
    #     return 1
    # ratioScore = len(detections) / stroma_area * 10000
    # print("RatioScore : %f" % (ratioScore))
    
    # if ratioScore < 1 :
    #     return 1
    # elif ratioScore > 8 :
    #     return 95
    # else :
    #     return int(94/7 * (ratioScore - 1) + 1)
    return 100

device = 'cuda:0'
WEIGHT_DIR = "/home/user/WEIGHT/segmentation"
# config_file_class_3 = osp.join(WEIGHT_DIR, 'config_class_3.py')
config_file_res101 = osp.join(WEIGHT_DIR, 'config_folder_1.py')
# config_file_swin = osp.join(WEIGHT_DIR, 'config_tumor_swin.py')

# checkpoint_file_class_3 = osp.join(WEIGHT_DIR, 'weight_class_3.pth')
checkpoint_file_res101_folder1 = osp.join(WEIGHT_DIR, 'weight_res101_folder1.pth')
checkpoint_file_res101_folder2 = osp.join(WEIGHT_DIR, 'weight_res101_folder2.pth')
# checkpoint_file_swin = osp.join(WEIGHT_DIR, 'weight_swin.pth')

## Detection init
DETECTION_WEIGHT_DIR = "/home/user/WEIGHT/detection"



print(f"Pytorch GPU available: {torch.cuda.is_available()}")

_bs_pred1_det = 72
_bs_pred2_det = 24
_bs_pred5_det = 24
_bs_pred6_det = 24
_bs_pred7_det = 24

def process():
    """Proceses a test slide"""

    level = READING_LEVEL
    tile_size = WRITING_TILE_SIZE # should be a power of 2
    #tile_size_bulk = 1024  
    
    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    print(f'Processing image: {image_path}')
    print(f'Processing with mask: {tissue_mask_path}')

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()
    # create writers
    print(f"Setting up writers")
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    print("Processing image...")
    # G_model_class_3 = init_segmentor(config_file_class_3, checkpoint_file_class_3, device=device)
    G_model_res101_folder1 = init_segmentor(config_file_res101, checkpoint_file_res101_folder1, device=device)
    G_model_res101_folder2 = init_segmentor(config_file_res101, checkpoint_file_res101_folder2, device=device)
    # G_model_swin = init_segmentor(config_file_swin, checkpoint_file_swin, device=device)
    G_model_swin = None

    cfg = mmcv.Config.fromfile(config_file_res101)
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    G_test_pipeline = Compose(test_pipeline)
    # loop over image and get tiles
    for y in range(0, dimensions[1], tile_size):
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()
            
            if not np.any(tissue_mask_tile):
                #print("{}, {}".format(x,y) )
                continue
            
            Padding = 512
            x_seg = x - Padding
            y_seg = y - Padding
            tile_size_seg = tile_size + 2*Padding
            
            image_tile_seg = image.getUCharPatch(
                startX=x_seg, startY=y_seg, width=tile_size_seg, height=tile_size_seg, level=level
            )
            # segmentation
            print("=========segmentation start=========")
            segmentation_mask_class3 = process_image_tile_to_segmentation(
                image_tile_seg=image_tile_seg, tissue_mask_tile=tissue_mask_tile, 
                model_res101_folder1=G_model_res101_folder1, 
                model_res101_folder2=G_model_res101_folder2, 
                model_swin = G_model_swin, 
                test_pipeline=G_test_pipeline,
                Padding=Padding)
            segmentation_writer.write_segmentation(tile=segmentation_mask_class3, x=x, y=y)
            
    print("Saving Segmentation...")
    # save segmentation and detection
    segmentation_writer.save()
    del G_model_res101_folder1
    del G_model_res101_folder2
    torch.cuda.empty_cache()
    torch.cuda.init()
    
    #### 256
    device_yolov5 = select_device(device)
    weights_yolov5 = osp.join(DETECTION_WEIGHT_DIR, 'best_tmp.pt')
    model_yolov5 = DetectMultiBackend(weights_yolov5, device=device_yolov5)
    torch.cuda.empty_cache()

    detection_config_file = osp.join(DETECTION_WEIGHT_DIR, 'cascade_1.py')
    detection_checkpoint_file = osp.join(DETECTION_WEIGHT_DIR, 'cascade_1.pth')
    model_mm = init_detector(detection_config_file, detection_checkpoint_file, device=device)
    torch.cuda.empty_cache() 
    #### 384
    weights_yolov5_2 = osp.join(DETECTION_WEIGHT_DIR, 'best_384.pt')
    model_yolov5_2 = DetectMultiBackend(weights_yolov5_2, device=device_yolov5)
    torch.cuda.empty_cache()

    detection_config_file4 = osp.join(DETECTION_WEIGHT_DIR, 'cascade_2_384.py')
    detection_checkpoint_file4 = osp.join(DETECTION_WEIGHT_DIR, 'cascade_2_384.pth')
    model_mm4 = init_detector(detection_config_file4, detection_checkpoint_file4, device=device)
    torch.cuda.empty_cache() 

    detection_config_file5 = osp.join(DETECTION_WEIGHT_DIR, 'atss_2_384.py')
    detection_checkpoint_file5 = osp.join(DETECTION_WEIGHT_DIR, 'atss_2_384.pth')
    model_mm5 = init_detector(detection_config_file5, detection_checkpoint_file5, device=device)
    torch.cuda.empty_cache() 
    
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)
    # Small Size Scanning image info
    downSample = 64
    level_smallMask = tissue_mask.getBestLevelForDownSample(downSample) 
    dimensions_smallMask = tissue_mask.getLevelDimensions(level_smallMask)
    smallMask = tissue_mask.getUCharPatch(startX=0, startY=0, width=dimensions_smallMask[0], height=dimensions_smallMask[1], level=level_smallMask).squeeze()
    
    kernel = np.ones((3,3), np.uint8)   
    smallMask = cv2.dilate(smallMask, kernel, iterations = 2)     
    smallMask = cv2.erode(smallMask, kernel, iterations = 2)
    
    mask_coor = []  
    for iH in range(0, smallMask.shape[0]) :
        for iW in range(0, smallMask.shape[1]) :
            if smallMask[iH][iW] != 0 :
                begin_x = iW
                begin_y = iH
                tmp = 1
                while(iW+tmp != smallMask.shape[1] and smallMask[iH][iW+tmp-1] != 0) :
                    tmp += 1
                end_x = iW+tmp-2
                tmp = 1
                while(iH+tmp != smallMask.shape[0] and smallMask[iH+tmp-1][iW] != 0) :
                    tmp += 1
                end_y = iH+tmp-2
                
                height_box = end_y - begin_y
                width_box = end_x - begin_x
                if width_box < (1024/downSample) :
                    width_box = int(1024/downSample) - 1
                if height_box < (1024/downSample) :
                    height_box = int(1024/downSample) - 1
                    
                smallMask[begin_y:begin_y + height_box + 1, begin_x:begin_x + width_box + 1] = 0
                mask_coor.append((begin_x, begin_y, begin_x + width_box + 1, begin_y + height_box + 1))
                
    for i in range(0, len(mask_coor)):
        x = mask_coor[i][0] * downSample - 256
        y = mask_coor[i][1] * downSample - 256
        
        width = (mask_coor[i][2] - mask_coor[i][0] + 1) * downSample + 512
        height = (mask_coor[i][3] - mask_coor[i][1] + 1) * downSample + 512
        width = int(math.floor(width/256)*256)
        height = int(math.floor(height/256)*256)
        if(width < 1024):
            width = 1024
        if(height < 1024):
            height = 1024            
    
        tissue_mask_tile = tissue_mask.getUCharPatch(
            startX=x, startY=y, width=width, height=height, level=level
        ).squeeze()

        if not np.any(tissue_mask_tile):
            #print("{}, {}".format(x,y) )
            continue

        image_tile = image.getUCharPatch(
            startX=x, startY=y, width=width, height=height, level=level
        )

        print("========= detection processing =========")
        # ts = time()
        detections = process_image_tile_to_detections(
            image_tile=image_tile,
            tissue_mask = tissue_mask_tile, 
            model1 = model_yolov5, 
            model2 = model_mm, 
            model5 = model_yolov5_2, 
            model6 = model_mm4, 
            model7 = model_mm5, 
            device = device
        )
        # te = time()
        # print("process_image_tile_to_detections took: %2.4f sec" % (te - ts))
        
        detection_writer.write_detections(
            detections=detections, spacing=spacing, x_offset=x, y_offset=y
        )

    print("Saving...")
    # save segmentation and detection
    detection_writer.save()

    print('Number of detections', len(detection_writer.detections))
    
    print("Compute tils score...")
    # compute tils score
    tils_score = process_segmentation_detection_to_tils_score(
        TMP_SEGMENTATION_OUTPUT_PATH, detection_writer.detections
    )
    tils_score_writer.set_tils_score(tils_score=tils_score)
    print("Tils Score : %f" % tils_score)

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")


# if __name__ == '__main__':
#     process()