## EfficientDet： Pytorch Implementation Support Odgt_Nori and MS COCO

**EfficientDet的pytorch版高精度实现：** 本项目属于codebase建设的一部分，旨在支持odgt格式的数据标注以及Nori的image存储格式，在不用改任何参数（包括数据路径）的情况下，也能一键跑通本项目代码。

对于odgt标注格式的数据，本项目支持两种读取方式：一种是将odgt标注文件和images放在本地路径下（以下统称为  **：Local** ），另一种是odgt标注文件存于OSS中，image是nori存储方式（以下统称为  **：OSS** ），下面对这两种方式分别进行说明：

### OSS

#### 数据

不需要本地存储

#### 参数设置

对应 daodixian_2class.yml

```
project_name: daodixian_2class  # 这个 project_name 既是 yml文件的名称 ，又是数据集的根目录（如果是local本地读取的话）

obj_list: ['songduangu','yiwu']   # 表示数据中的class name

set_class: daodixian_2class  # 仅在 Local 模式时需要设置这个参数 ，表示odgt标注文件的前缀，后缀为_train.odgt（固定的）

train_odgt_path : 's3://jiashuaishuai/dianwang_data/daodixian_2class/annotations/daodixian_2class_train.odgt'  # 根据odgt和其中的nori_id来读取标注和Image

val_odgt_path : 's3://jiashuaishuai/dianwang_data/daodixian_2class/annotations/daodixian_2class_val.odgt'

注：还有部分参数在训练脚本里的argparse进行更改
```



#### 训练脚本

对应train_odgt_nori_daodixian.py

注：train_coco_daodixian.py 是用来训练json标注格式的daodixian数据的

#### 评估脚本

对应eval_odgt_nori.py

### Local

#### 数据

```
# 你的数据结构应该按照以下格式：（以导地线odgt为例，json也同理）

datasets/
    -daodixian_2class/
        -images/
            -*.jpg
        -annotations
            - daodixian_2class_train.odgt
            - daodixian_2class_val.odgt
注：images中存训练和验证的所有images
daodixian_2class就是daodixian_2class.yml 中的set_class参数，以 _train.odgt 和_val.odgt 结尾是固定的
```

#### 参数设置

同OSS模式下的参数设置

#### 训练脚本

对应train_odgt_daodixian.py

#### 评估脚本

对应eval_odgt.py

### Pretrained weights and benchmark

The performance in COCO datasets is very close to the paper's, it is still SOTA.

The speed/FPS test includes the time of post-processing with no jit/data precision trick.

| coefficient | pth_download | GPU Mem(MB) | FPS | Extreme FPS (Batchsize 32) | mAP 0.5:0.95(this repo) | mAP 0.5:0.95(official) |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| D0 | [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) | 1049 | 36.20 | 163.14 | 33.1 | 33.8
| D1 | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth) | 1159 | 29.69 | 63.08 | 38.8 | 39.6
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 1321 | 26.50 | 40.99 | 42.1 | 43.0
| D3 | [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) | 1647 | 22.73 | - | 45.6 | 45.8
| D4 | [efficientdet-d4.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth) | 1903 | 14.75 | - | 48.8 | 49.4
| D5 | [efficientdet-d5.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth) | 2255 | 7.11 | - | 50.2 | 50.7
| D6 | [efficientdet-d6.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth) | 2985 | 5.30 | - | 50.7 | 51.7
| D7 | [efficientdet-d7.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d7.pth) | 3819 | 3.73 | - | 52.7 | 53.7
| D7X | [efficientdet-d8.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d8.pth) | 3983 | 2.39 | - | 53.9 | 55.1

### Install

    # install requirements
    pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0

### Reference

本项目代码参考的是 [zylo117](https://github.com/zylo117)  的 **[Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)**  ，其中 readme_ori.md 就是该作者的原readme，使用本项目代码之前，建议先看一下readme_ori.md

### Contact

ssjia_cv@foxmail.com


