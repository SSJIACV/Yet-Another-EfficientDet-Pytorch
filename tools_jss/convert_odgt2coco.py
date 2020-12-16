"""
This python script is used for 将odgt标注转换为coco格式的json标注
"""
# @Time    : 2020/10/26 12:01
# @Author  : jss
# @email   : ssjia_cv@foxmail.com
# @File    : convert_odgt2coco.py

import os
import glob
import json
from tqdm import tqdm
import shutil
import numpy as np

# # 注意：这个转换程序会丢失odgt中的nori id信息

# def convert(odgt_file1, odgt_file2, json_file):
def convert(odgt_file1, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = 1  # gtbox起始Id
    image_id_start = 20201000001  # image起始id
    all_categories = {}
    with open(odgt_file1, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]  # str to list
    pbar = tqdm(total=len(records))
    image_id = 0
    for index, record in enumerate(records):
        # pbar.update(1)
        filename = record['fpath']
        image_id = image_id_start + index
        width = int(record['width'])
        height = int(record['height'])
        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
        # json_dict['images'].append(image)
        flag = 0 # the image is zero annotations?
        for gtbox in record['gtboxes']:
            box = gtbox['box']
            xmin, ymin, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            category = gtbox['tag']
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            if category not in categories:
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories) + 1
                print(
                    "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
                        category, pre_define_categories, new_id))
                categories[category] = new_id
            category_id = categories[category]
            ann = {'area': w*h, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, w, h],
                   'category_id': category_id, 'id': bnd_id}
            json_dict['annotations'].append(ann)
            flag = 1
            bnd_id = bnd_id + 1
        if flag==1:
            json_dict['images'].append(image)
            pbar.update(1)


    for cate, cid in categories.items():
        cat = {'supercategory': cate, 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories),
                                                                                  all_categories.keys(),
                                                                                  len(pre_define_categories),
                                                                                  pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())

if __name__ == '__main__':
    # 定义你自己的类别
    # classes = ['good', 'zangwu']
    # classes = ['fangzhenchui_good', 'fangzhenchui_bad']
    # classes = ['xianjia_good', 'xianjia_bad']
    # classes = ['zhongchui_good', 'zhongchui_bad']
    # classes = ['songduangu', 'yiwu']
    # classes = [
    #     'normal_board',  # 1
    #     'bad_board',  # 2
    #     'normal_driving_birds_device', # 3
    #     'bad_driving_birds_device', # 4
    # ]
    # classes = ['ganta', ]
    # classes = ['tongdao', ]
    classes = ['jichu', ]
    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
    # 这里也可以自定义类别id，把上面的注释掉换成下面这行即可
    # pre_define_categories = { '1': 1, '2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,'10': 10}
    only_care_pre_define_categories = True  # or False 只生成想要的类别
    odgt_input1 = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/tools_jss/odgt_files_202006/jichu_1class_val.odgt'
    # odgt_input2 = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/tools_jss/odgt_files_202006/daodixian_2class.odgt'
    # odgt_input = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/tools_jss/odgt_files_1121/dajinju_val.odgt'
    save_json = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/tools_jss/json_files/jichu_1class_val.json'
    # convert(odgt_input1,odgt_input2,save_json)
    convert(odgt_input1,save_json)
    print('convert done')
