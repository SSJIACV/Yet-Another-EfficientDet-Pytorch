#!/usr/bin/env python3
import os
import json
import cv2
import sys
from copy import deepcopy
from collections import defaultdict
from IPython import embed

def coco_json2odgt(jsonpath,save_odgtpath):

    with open(jsonpath) as f:
        json_gt = json.load(f)

    image_info_list = json_gt['images']
    print("num of gt", len(image_info_list))
    image_id_dict = dict()
    len_img = len(image_info_list)

    category2name = dict()
    for cat in json_gt['categories']:
        category2name[cat['id']] = cat['name']

    image2annotlist = defaultdict(list)
    for annot in json_gt['annotations']:
        image_id = annot['image_id']
        image2annotlist[image_id].append(annot)
    print('colloct down')


    for idx, item in enumerate(image_info_list):
        # print('idx', idx, 'num', len_img)
        image_id = item['id']
        image_id_dict[image_id] = item

    print('start down')
    odgt_lines = []
    for image_id in image_id_dict.keys():
        image_info = image_id_dict[image_id]
        annot_list = image2annotlist[image_id]
        odgtline = dict()
        odgtline['fpath'] = image_info['file_name']
        odgtline['height'] = image_info['height']
        odgtline['width'] = image_info['width']
        # odgtline['ID'] = image_info['ID']
        gtbox_list = list()
        for annot in annot_list:
            #if annot['category_id'] not in [2, 3, 4]:
            #    continue 
            gtbox_item = dict()
            gtbox_item['box'] = annot['bbox']
            cat_id = annot['category_id']
            gtbox_item['tag'] = category2name[cat_id]
            # print(gtbox_item['tag'])
            gtbox_list.append(gtbox_item)
        odgtline['gtboxes'] = gtbox_list
        odgt_lines.append(json.dumps(odgtline) + '\n')

    with open(save_odgtpath, 'w') as f:
        f.writelines(odgt_lines)



if __name__ == '__main__':

    jsonpath = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/datasets/ganta_1class/annotations/ganta_1class_val.json'
    save_odgtpath = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/datasets/ganta_1class/annotations/ganta_1class_val.odgt'
    coco_json2odgt(jsonpath,save_odgtpath)
    print('done')