"""
This python script is used for 将odgt转换为需要的tag
"""
# @Time    : 2020/10/26 11:21
# @Author  : jss
# @email   : ssjia_cv@foxmail.com
# @File    : convert_odgt_to_new_tag.py

import json
from tqdm import tqdm
input_file = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/tools_jss/odgt_files_2019/ganta_val.odgt'  # jueyuanzi_det1_train.odgt
# input_file_1 = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/tools_jss/odgt_files_202006/daodixian_2class.odgt'
output_file = '/data/wurenji/code_new/Yet-Another-EfficientDet-Pytorch/tools_jss/odgt_files_2019/ganta_1class_val.odgt'
# code = ['fangzhenchui_good','fangzhenchui_bad']
# code = ['xianjia_good','xianjia_bad']
# code = ['zhongchui_good','zhongchui_bad']
code = ['songduangu','yiwu']
# code_good = [
#     '030000000',
#     '030100000',
#     '030200000',
# ]
# code_zangwu = [
#     '030000011',
#     '030100011',
#     '030200041',
# ]

# code_songduangu = ['020001073','020001011','020001031','020000011','020000012','020000031','020100012']
# code_yiwu = ['020001061','020000111','020100051']
def transform():
    all_results = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]  # str to list
    
    pbar = tqdm(total=len(records))
    for record in records:
        # pbar.update(1)
        new_item = {}
        # new_item['raw_fpath_list'] = record['raw_fpath_list']
        new_item['fpath'] = record['fpath']
        new_item['height'] = record['height']
        new_item['width'] = record['width']
        new_item['ID'] = record['ID']
        new_gtboxes = []
        for gtbox in record['gtboxes']:
            old_tag = gtbox['tag']
            new_tag = 'ganta'
            new_gtboxes.append({'tag': new_tag, 'box': gtbox['box']})
            # if old_tag in code_songduangu:
            #     new_tag = 'songduangu'
            #     new_gtboxes.append({'tag': new_tag, 'box': gtbox['box']})
            # if old_tag in code_yiwu:
            #     new_tag = 'yiwu'
            #     new_gtboxes.append({'tag': new_tag, 'box': gtbox['box']})
            # if old_tag in code:
            #     new_gtboxes.append({'tag': old_tag, 'box': gtbox['box']})
        if len(new_gtboxes)!=0:
            pbar.update(1)
            new_item['gtboxes'] = new_gtboxes
            all_results.append(json.dumps(new_item))

    # with open(input_file_1, 'r', encoding='utf-8') as f_in:
    #     lines = f_in.readlines()
    # records = [json.loads(line.strip('\n')) for line in lines]  # str to list
    
    # pbar = tqdm(total=len(records))
    # for record in records:
    #     # pbar.update(1)
    #     new_item = {}
    #     # new_item['raw_fpath_list'] = record['raw_fpath_list']
    #     new_item['fpath'] = record['fpath']
    #     new_item['height'] = record['height']
    #     new_item['width'] = record['width']
    #     new_item['ID'] = record['ID']
    #     new_gtboxes = []
    #     for gtbox in record['gtboxes']:
    #         old_tag = gtbox['tag']
    #         # new_tag = 'tongdao'
    #         # new_gtboxes.append({'tag': new_tag, 'box': gtbox['box']})
    #         # if old_tag in code_songduangu:
    #         #     new_tag = 'songduangu'
    #         #     new_gtboxes.append({'tag': new_tag, 'box': gtbox['box']})
    #         # if old_tag in code_yiwu:
    #         #     new_tag = 'yiwu'
    #         #     new_gtboxes.append({'tag': new_tag, 'box': gtbox['box']})
    #         if old_tag in code:
    #             new_gtboxes.append({'tag': old_tag, 'box': gtbox['box']})
    #     if len(new_gtboxes)!=0:
    #         pbar.update(1)
    #         new_item['gtboxes'] = new_gtboxes
    #         all_results.append(json.dumps(new_item))

    fw = open(output_file, 'w')
    for res in all_results:
        fw.write(res + '\n')

    fw.close()

if __name__ == "__main__":
    transform()
    print('done')