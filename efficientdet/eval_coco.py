# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

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

# rlaunch --cpu=8 --gpu=1 --memory=$((80*1024)) --max-wait-time 10h --preemptible=no --charged-group v_tracking -- python3 coco_eval.py

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='daodixian_2class', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=3, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default='/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/logs/dajinju_xianjia_d3/dajinju/efficientdet-d3_70_19312.pth', help='/path/to/weights')
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
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

det_save_json = f'dajinju_xianjia_2class_bbox_results_d3_epoch70_1125.json'

def evaluate_coco_show_res_jss(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    count = 0
    for image_id in tqdm(image_ids):
        count = count + 1
        if count > 21:
            break
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']
        print('image path:',image_path)

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

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }
                score = float(score)
                category_id = label + 1
                box = box.tolist()
                # print('box:',box)
                xmin, ymin, w, h, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), score
                if score > 0.2:
                    cv2.rectangle(ori_imgs[0], (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 6)
                    cv2.putText(ori_imgs[0], '{}:{:.2f}'.format(category_id, score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 6)
                results.append(image_result)
        cv2.imwrite('./test_result/zhongchui_d3_epoch200_1124/'+'tmp'+'{}'.format(count)+'.jpeg',ori_imgs[0])
    

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    # filepath = f'{set_name}_bbox_results.json'
    filepath = det_save_json
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']
        # print('image path:',image_path)

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

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }
                results.append(image_result)
    

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    # filepath = f'{set_name}_bbox_results.json'
    filepath = det_save_json
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    # only test yiwu
    # coco_eval.params.catIds = [2]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    SET_NAME = params['val_set']
    VAL_IMGS = f'datasets/daodixian_2class/images/'
    VAL_GT = f'datasets/daodixian_2class/annotations/daodixian_2class_val.json'
    # VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    # VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/' # /data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/datasets/coco/val2014
    # VAL_IMGS = f'datasets//coco/val2014/' # 这句是用来验证minival时
    MAX_IMAGES = 1000000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    
    if not os.path.exists(det_save_json):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    _eval(coco_gt, image_ids, det_save_json)
