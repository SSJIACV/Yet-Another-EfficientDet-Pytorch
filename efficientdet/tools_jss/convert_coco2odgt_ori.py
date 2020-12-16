#!/usr/bin/env python3
import os
import json
import cv2
import sys
from copy import deepcopy
from collections import defaultdict
from IPython import embed


def parsejson2odgt(jsonpath, image_dir, saveodgtpath, norisavepath=''):

    #write_nori = norisavepath
    write_nori = False

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

    if write_nori:
        import nori2 as nori
        nw = nori.open(norisavepath, 'w')

    for idx, item in enumerate(image_info_list):
        print('idx', idx, 'num', len_img)
        filename = item['file_name']
        image_path = os.path.join(image_dir, filename)
        assert os.path.exists(image_path), 'file path does not exists'
        if write_nori:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            d = cv2.imencode('.bmp', image)[1].tostring()
            data_id = nw.put(d)
            item['nori_id'] = data_id
        image_id = item['id']
        image_id_dict[image_id] = item

    if write_nori:
        nw.close()
        print("save nori down")
        print("---------------------")

    print('start down')
    odgt_lines = []
    for image_id in image_id_dict.keys():
        image_info = image_id_dict[image_id]
        annot_list = image2annotlist[image_id]
        odgtline = dict()
        odgtline['fpath'] = image_info['file_name']
        odgtline['image_info'] = image_info
        odgtline['height'] = image_info['height']
        odgtline['width'] = image_info['width']
        #odgtline['nori_id'] = image_info['nori_id']
        gtbox_list = list()
        for annot in annot_list:
            #if annot['category_id'] not in [2, 3, 4]:
            #    continue 
            gtbox_item = dict()
            gtbox_item['extra'] = annot
            gtbox_item['extra']["category_id"] = 1
            gtbox_item['box'] = annot['bbox']
            cat_id = annot['category_id']
            gtbox_item['tag'] = 1
            gtbox_list.append(gtbox_item)
        odgtline['gtboxes'] = gtbox_list
        odgt_lines.append(json.dumps(odgtline) + '\n')

    with open(saveodgtpath, 'w') as f:
        f.writelines(odgt_lines)


if __name__ == '__main__':
    assert len(sys.argv) == 5
    parsejson2odgt(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
# vim: ts=4 sw=4 sts=4 expandtab
