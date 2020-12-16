"""
This python script is used for 计算model的flops
"""
# @Time    : 2020/12/01 20:24
# @Author  : jss
# @email   : ssjia_cv@foxmail.com
# @File    : flops_count.py

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

from ptflops import get_model_complexity_info

# rlaunch --cpu=8 --gpu=1 --memory=$((80*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 flops_count.py

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='fushusheshi_4class', help='project file that contains parameters')
# ap.add_argument('-p', '--project', type=str, default='jueyuanzi_zangwu_2class', help='project file that contains parameters')
# ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=3, help='coefficients of efficientdet')
# ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('-w', '--weights', type=str, default='/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/logs/fushusheshi_4class_d3_lr_1e-3/fushusheshi_4class/efficientdet-d3_260_15138.pth', help='/path/to/weights')
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

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
# obj_list = params['obj_list']
# obj_list = params['obj_list_fangzhenchui']
# obj_list = params['obj_list_xianjia']
obj_list = params['obj_list_fushusheshi']
# obj_list = params['obj_list_zhongchui']
# obj_list = params['obj_list_daodixian']
# obj_list = params['obj_list_tongdao']
# obj_list = params['obj_list_ganta']


# input_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1536, 1536]
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
# input_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 768, 1536]

det_save_odgt = f'val_result/fushusheshi_4class_d3_lr_1e-3_only_bad_epoch260_1201.odgt'
det_save_eval_log_txt = f'val_result/fushusheshi_4class_d3_lr_1e-3_only_bad_epoch260_1201.txt'
threshold=0.05




if __name__ == '__main__':

    with torch.cuda.device(0):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                        ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


