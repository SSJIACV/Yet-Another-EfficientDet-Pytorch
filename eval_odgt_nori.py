"""
This python script is used for 支持odgt 和  nori格式的验证
"""
# @Time    : 2020/12/15 
# @Author  : jss
# @email   : ssjia_cv@foxmail.com
# @File    : eval_odgt_nori.py

import json
import numpy as np
import os
import cv2
import argparse
import torch
from torch import nn
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from utils.sync_batchnorm import convert_model

from refile import smart_open
import nori2 as nori
from meghair.utils.imgproc import imdecode
nf = nori.Fetcher()

# rlaunch --cpu=8 --gpu=1 --memory=$((80*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 eval_odgt_nori.py

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='daodixian_2class', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=3, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default='/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/logs/daodixian_2class_d3_odgt_1215/daodixian_2class/efficientdet-d3_50_17238.pth', help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=False, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

print(f'running odgt_nori_style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))

obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

val_odgt_path = 's3://jiashuaishuai/dianwang_data/daodixian_2class/annotations/daodixian_2class_val.odgt'

det_save_odgt = f'val_result/daodixian_2class_d3_lr_1e-3_epoch50_1215_nori_test.odgt'
det_save_eval_log_txt = f'val_result/daodixian_2class_d3_lr_1e-3_epoch50_1215_nori_test.txt'


threshold=0.05


def evaluate_odgt(records, model):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    all_result = []
    pbar = tqdm(total=len(records))
    for record in records:
        pbar.update(1)
        nori_ids = record['ID']
        # img = imdecode(nf.get(nori_id))
        ori_imgs, framed_imgs, framed_metas = preprocess_nori(nori_ids, max_size=input_sizes[compound_coef])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        dtboxes = []
        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores
            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                # category_id = label + 1
                category_id = label
                tag = obj_list[category_id]
                # only test bad 
                # if tag != 'zangwu':
                #     continue
                # if tag in ['normal_board', 'normal_driving_birds_device']:
                #     continue
                box = rois[roi_id, :]
                box = box.tolist()
                box_new = dict(box=box, score=score, tag=tag)
                dtboxes.append(box_new)
        record['dtboxes'] = dtboxes
        
        # new_gtboxes = []
        # for gtbox in record['gtboxes']:
        #     # if gtbox['tag'] == 'fangzhenchui_bad' or gtbox['tag'] == 'fangzhenchui_good':
        #     #     new_gtboxes.append(gtbox)
        #     if gtbox['tag'] == 'zangwu':
        #         new_gtboxes.append(gtbox)
        #     # if gtbox['tag'] in ['normal_board', 'normal_driving_birds_device']:
        #     #     continue
        #     # else:
        #     #     new_gtboxes.append(gtbox)
        # record['gtboxes'] = new_gtboxes
        all_result.append(record)

    fw = open(det_save_odgt, 'w')
    for res in all_result:
        res = json.dumps(res)
        fw.write(res + '\n')
    fw.close()

    # evaluation
    eval_script = '/data/wurenji/code_new/dianwang_detection/evalTookits2/eval.py'
    command = 'python3 -u %s --dt=%s --gt=%s --iou=%f | tee -a %s' % (eval_script, det_save_odgt, det_save_odgt, 0.2, det_save_eval_log_txt)
    os.system(command)
    print('done')

def preprocess_nori(*nori_ids, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [imdecode(nf.get(nori_id))[..., ::-1] for nori_id in nori_ids]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas
    

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,

if __name__ == '__main__':
    if os.path.exists(det_save_odgt):
        eval_script = '/data/wurenji/code_new/dianwang_detection/evalTookits2/eval.py'
        command = 'python3 -u %s --dt=%s --gt=%s --iou=%f | tee -a %s' % (eval_script, det_save_odgt, det_save_odgt, 0.2, det_save_eval_log_txt)
        os.system(command)
        print('done')
    else:
        VAL_GT = val_odgt_path
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                        ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()
        if use_cuda:
            model.cuda(gpu)
            if use_float16:
                model.half()
        with smart_open(VAL_GT, 'r') as f:
            lines = [l.rstrip() for l in f.readlines()]
        records = [json.loads(line) for line in lines]  # str to list
        evaluate_odgt(records, model)
