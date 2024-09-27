# -*- coding: utf-8 -*-



from os.path import join
from os import listdir
from torch.utils.data import Dataset
from PIL import Image

import torch
import torch.nn.functional as F
from functools import reduce
import numpy as np
import os


from torch.utils.data import DataLoader
import os, sys

import monai
from monai.data import decollate_batch, PILReader,NumpyReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImage,
    SpatialPad,
    RandSpatialCrop,
    RandRotate90,
    ScaleIntensityd,
    RandAxisFlip,
    RandZoom,
    RandGaussianNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
    RandHistogramShift,
    EnsureType,
    ResizeWithPadOrCrop,
)

from glob import glob
import tifffile
import scipy.ndimage as ndi

###########################

#�ļ�Ŀ¼��../../dataset/light 

class MyDataset(Dataset):
    def __init__(self, db_path, transform=None):
        types = glob(os.path.join(db_path, '*'))
        image=[]
        seg=[]
        self.transform = transform
        for type in types:
            now_imgs = glob(os.path.join(type, 'img', '*.tiff'))
            image.extend(now_imgs)
            now_lbls = glob(os.path.join(type, 'mask', '*mask.tiff'))
            seg.extend(now_lbls)

        self.images = sorted(image)
        self.segs = sorted(seg)
        

    
    def __len__(self):
        return len(self.images)               
                    
    def __getitem__(self, index):

        input_image_path = self.images[index]
        output_image_path = self.segs[index]
        
        

        if 'NEP25' in input_image_path:
            lv = 1
        else:
            lv = 2

        img_tiff = tifffile.imread(input_image_path,key=0)
        img_tiff_X20 = ndi.zoom(img_tiff, (1/lv, 1/lv, 1), order=1)
        
        out_tiff = tifffile.imread(output_image_path,key=0)
        out_tiff_X20 = ndi.zoom(out_tiff, (1 / lv, 1 / lv), order=1)
        
        temp={"input":img_tiff_X20,"output":out_tiff_X20}

        if self.transform:
            temp = self.transform(temp)

        input_image = temp["input"]
        output_image = temp["output"]   
        return input_image, output_image ,input_image_path
            
    

    def __len__(self):
        return len(self.images)




def main():



    train_transforms = Compose([EnsureChannelFirstd(keys=["input"],channel_dim='2'),
                                EnsureChannelFirstd(keys=["output"],channel_dim='no_channel'),
                                ScaleIntensityd(keys=["input","output"])])
    
    
    dataset = MyDataset("/ssd/cyj/KPIs24datanew/task2/val/",transform=train_transforms)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (inputs, labels,patch) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"Inputs size: {inputs.size()}")
        print(f"Labels size: {labels.size()}")
        print(f"path: {patch}")

if __name__ == "__main__":
    main()
