import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import json
from refile import smart_open
import nori2 as nori
from meghair.utils.imgproc import imdecode
nf = nori.Fetcher()


class Odgt_nori_Dataset(Dataset):  
    def __init__(self, set='train',train_odgt = 's3://jiashuaishuai/dianwang_data/daodixian_2class/annotations/daodixian_2class_train.odgt', val_odgt = 's3://jiashuaishuai/dianwang_data/daodixian_2class/annotations/daodixian_2class_val.odgt', CLASSES_NAME = ["songduangu","yiwu"], transform=None):

        self.transform = transform
        self.name2id=dict(zip(CLASSES_NAME,range(len(CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}

        if set == 'train':
            self.odgt = train_odgt
        else:
            self.odgt = val_odgt
        with smart_open(self.odgt, 'r') as f:
            lines = [l.rstrip() for l in f.readlines()]
        records = [json.loads(line) for line in lines]  # str to list
        self.image_ids = records

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.image_ids[image_index]
        nori_id = image_info.get('ID', None)
        img = imdecode(nf.get(nori_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        gtboxes = self.image_ids[image_index]['gtboxes']
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(gtboxes) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(gtboxes):

            # some gtboxes have basically no width / height, skip them
            if a['box'][2] < 1 or a['box'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['box']
            annotation[0, 4] = self.name2id[a['tag']]
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


class OdgtDataset(Dataset):  
    def __init__(self, root_dir, set='train',set_class = 'daodixian_2class', CLASSES_NAME = ["songduangu","yiwu"], transform=None):

        self.root_dir = root_dir
        self.img_dir = 'images'
        self.transform = transform
        self.set_class = set_class
        self.name2id=dict(zip(CLASSES_NAME,range(len(CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}

        if set == 'train':
            self._annopath = os.path.join(self.root_dir, "annotations", self.set_class + '_train.odgt')
        else:
            self._annopath = os.path.join(self.root_dir, "annotations", self.set_class + '_val.odgt')
        with open(self._annopath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]  # str to list
        self.image_ids = records

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.image_ids[image_index]
        path = os.path.join(self.root_dir, self.img_dir, image_info['fpath'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        gtboxes = self.image_ids[image_index]['gtboxes']
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(gtboxes) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(gtboxes):

            # some gtboxes have basically no width / height, skip them
            if a['box'][2] < 1 or a['box'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['box']
            annotation[0, 4] = self.name2id[a['tag']]
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train',set_class = 'fangzhenchui_2class', transform=None):

        self.root_dir = root_dir
        self.img_dir = 'images'
        self.transform = transform

        # self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.img_dir + '.json'))
        if set == 'train':
            self.coco = COCO(os.path.join(self.root_dir, 'annotations', set_class + '_train.json'))
        else:
            self.coco = COCO(os.path.join(self.root_dir, 'annotations', set_class + '_val.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.img_dir, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
