# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import paddle
import numpy as np
from PIL import Image
import glob
from .transforms.transforms import Compose
from .transforms import functional as F
from .camvid_cityscape import c2c

class Dataset(paddle.io.Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_path (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
            The contents of train_path file are as follow:
            image1.jpg ground_truth1.png
            image2.jpg ground_truth2.png
        val_path (str. optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        test_path (str, optional): The test dataset file. When mode is 'test', test_path is necessary.
            The annotation file is not necessary in test_path file.
        separator (str, optional): The separator of dataset list. Default: ' '.
        edge (bool, optional): Whether to compute edge while training. Default: False

        Examples:

            import paddleseg.transforms as T
            from paddleseg.datasets import Dataset

            transforms = [T.RandomPaddingCrop(crop_size=(512,512)), T.Normalize()]
            dataset_root = 'dataset_root_path'
            train_path = 'train_path'
            num_classes = 2
            dataset = Dataset(transforms = transforms,
                              dataset_root = dataset_root,
                              num_classes = 2,
                              train_path = train_path,
                              mode = 'train')

    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 num_classes=4,
                 mode='train',
                 dataset = 'VSPW',
                 clip = '01TP',
                 train_path='train.txt',
                 val_path='val.txt',
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.edge = edge
        self.dataset = dataset

        if mode.lower() not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.dataset_root = dataset_root

        if mode == 'train':
            file_path = train_path
        elif mode == 'val':
            file_path = val_path
        else:
            file_path = test_path
        if self.dataset =='cityscape':
            with open(file_path, 'r') as f:
                for line in f:
                    items = line.strip().split(separator)
                    if len(items) != 2:
                        if mode == 'train' or mode == 'val':
                            raise ValueError(
                                "File list format incorrect! In training or evaluation task it should be"
                                " image_name{}label_name\\n".format(separator))
                        image_path = os.path.join(self.dataset_root, items[0])
                        label_path = None
                    else:
                        image_path = os.path.join(self.dataset_root, items[0])
                        label_path = os.path.join(self.dataset_root, items[1])
                    self.file_list.append([image_path, label_path])
        elif self.dataset =='camvid':
            if (self.mode == 'train'):
                image_path = glob.glob(dataset_root+'/images/'+clip+'/'+'*.png')
                label_path = glob.glob(dataset_root+'/labels/'+clip+'/'+'*.png')
                image_path = sorted(image_path, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0][1:]))
                label_path = sorted(label_path, key=lambda x: int(x.split('/')[-1].split('_')[1][1:]))
                for i in range(len(image_path)):
                    self.file_list.append([image_path[i], label_path[i]])
            elif (self.mode == 'val'):
                image_path = glob.glob(dataset_root+'/images/'+clip+'_val/'+'*.png')
                label_path = glob.glob(dataset_root+'/labels/'+clip+'_val/'+'*.png')
                image_path = sorted(image_path, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0][1:]))
                label_path = sorted(label_path, key=lambda x: int(x.split('/')[-1].split('_')[1][1:]))
                for i in range(len(image_path)):
                    self.file_list.append([image_path[i], label_path[i]])
                self.file_list = self.file_list[:20]
        elif self.dataset =='VSPW':
            self.clip_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    self.clip_list.append(line)
            if(self.mode == 'val'):
                Num = len(self.clip_list)
                self.clip_list = self.clip_list[int(Num/2):]
                #self.clip_list = self.clip_list[:1]
            self.clip_img = []
            self.clip_label = []
            root_path = 'data/data104057/VSPW/data/'
            for clip in self.clip_list:
                img = glob.glob(root_path+clip[:-1]+'/origin/'+'*.jpg')
                img = [n for n in img if int(n.split('/')[-1].split('.')[0])%2==0]
                image_path = sorted(img, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                label = glob.glob(root_path+clip[:-1]+'/mask/'+'*.png')
                label = [n for n in label if int(n.split('/')[-1].split('.')[0])%2==0]
                label_path = sorted(label, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                if(len(image_path)%2 == 1):
                    image_path = image_path[:-1]
                if(len(label_path)%2 == 1):
                    label_path = label_path[:-1]
                self.clip_img = self.clip_img + image_path
                self.clip_label = self.clip_label + label_path
                self.clip_img.append('end')
                self.clip_label.append(clip)
                self.clip_img.append('end')
                self.clip_label.append(clip)

    def vspw_trans(self,label,mode):
        if(mode == 'val'):
            res = copy.deepcopy(label)
        else:
            res = label
        #add = [49, 61, 19, 15, 22, 23]
        add = [61]
        for i in range(len(add)):
            res[res == i] = 1
            res[res == add[i]] = i
        res[res>0] = 1
        return res

    def __getitem__(self, idx):
        if(self.dataset =='VSPW'):
            image_path = self.clip_img[idx]
            label_path = self.clip_label[idx]
            #print(image_path, label_path)
        else:
            image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            if(image_path == 'end'):
                return 1, 1
            im, _ = self.transforms(im=image_path)
            label = np.asarray(Image.open(label_path))
            label = label[np.newaxis, :, :]
            if(self.dataset == 'camvid'):
                label = c2c(label, self.mode)
            if(self.dataset == 'VSPW'):
                label = self.vspw_trans(label, self.mode)
            return im, label, image_path
        else:
            if(image_path == 'end'):
                return 1, 1
            im, label = self.transforms(im=image_path, label=label_path)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im, label, edge_mask
            else:
                if(self.dataset == 'camvid'):
                    label = c2c(label, self.mode)
                if(self.dataset == 'VSPW'):
                    label = self.vspw_trans(label, self.mode)
                return im, label, image_path

    def __len__(self):
        if(self.dataset =='VSPW'):
            return len(self.clip_img)
        else:
            return len(self.file_list)
