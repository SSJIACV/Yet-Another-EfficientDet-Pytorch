"""
This python script is used for 将网络输出转换为odgt形式,与RetinaNET对比时可以用同一个eval_mAP脚本
"""
# @Time    : 2020/11/24 20:42
# @Author  : jss
# @email   : ssjia_cv@foxmail.com
# @File    : eval_odgt.py

import json
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

# rlaunch --cpu=8 --gpu=1 --memory=$((80*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 eval_odgt.py

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

print(f'running odgt-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))

obj_list = params['obj_list]


input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

img_dir = './datasets/daodixian_2class/images/'
val_odgt_path = './datasets/daodixian_2class/annotations/daodixian_2class_val.odgt'

det_save_odgt = f'val_result/daodixian_2class_d3_lr_1e-3_epoch50_1215.odgt'
det_save_eval_log_txt = f'val_result/daodixian_2class_d3_lr_1e-3_epoch50_1215.txt'
threshold=0.05

def evaluate_odgt(VAL_IMGS, records, model):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    all_result = []
    pbar = tqdm(total=len(records))
    for record in records:
        pbar.update(1)
        fpath = record['fpath']
        image_path = os.path.join(VAL_IMGS,fpath)
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef])
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

    
if __name__ == '__main__':
    if os.path.exists(det_save_odgt):
        eval_script = '/data/wurenji/code_new/dianwang_detection/evalTookits2/eval.py'
        command = 'python3 -u %s --dt=%s --gt=%s --iou=%f | tee -a %s' % (eval_script, det_save_odgt, det_save_odgt, 0.2, det_save_eval_log_txt)
        os.system(command)
        print('done')
    else:
        VAL_IMGS = img_dir
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
        with open(VAL_GT, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]  # str to list
        evaluate_odgt(VAL_IMGS, records, model)

