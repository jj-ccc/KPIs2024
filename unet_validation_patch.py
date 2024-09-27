# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob
from net_sam_infer import  CellViTSAM
import torch.nn.functional as F

import torch
from PIL import Image

from monai import config
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity, Resize
from matplotlib import cm
import matplotlib.pyplot as plt


def save_validate(val_images, val_labels, val_outputs,output_dir, images, cnt):
    for i in range(val_images.shape[0]):
        folder_list = os.path.dirname(images[cnt+i]).split('/')
        save_folder = os.path.join(output_dir, folder_list[-3], folder_list[-2])

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        now_image = val_images[i].permute([2,1,0]).detach().cpu().numpy()
        now_label = val_labels[i][0].permute([1,0]).detach().cpu().numpy()
        now_pred = val_outputs[i][0].permute([1,0]).detach().cpu().numpy()
        name = os.path.basename(images[cnt+i])[0:-8]
        # plt.imsave(os.path.join(save_folder, 'val_%s_img.png' % (name)), now_image)
        # plt.imsave(os.path.join(save_folder, 'val_%s_lbl.png' % (name)), now_label, cmap = cm.gray)
        plt.imsave(os.path.join(save_folder, 'val_%s_mask.png' % (name)), now_pred, cmap = cm.gray)

    cnt += val_images.shape[0]
    return cnt

def main(tempdir, model_dir1,model_dir2, output_dir):
#def main(tempdir, model_dir1,model_dir2,model_dir3,model_dir4, output_dir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    image = []
    seg = []
    types = glob(os.path.join(tempdir, '*'))
    for type in types:
        cases = glob(os.path.join(type, '*'))
        for case in cases:
            now_imgs = glob(os.path.join(case, 'img', '*img.jpg'))
            image.extend(now_imgs)
            now_lbls = glob(os.path.join(case, 'mask', '*mask.jpg'))
            seg.extend(now_lbls)

    images = sorted(image)
    segs = sorted(seg)

    print('total image: %d' % (len(images)))

    # define transforms for image and segmentation
    imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    #imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity(), Resize(spatial_size=(512, 512), mode='nearest')])
    segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    outputrans = Compose([Resize(spatial_size=(2048, 2048), mode='nearest')])
    val_ds = ArrayDataset(images, imtrans, segs, segtrans)
    # sliding window inference for one image at every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    saver = SaveImage(output_dir=os.path.join(output_dir, "./validation_output"), output_ext=".png", output_postfix="seg")
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    model1 = CellViTSAM(input_classes=3,oputput_num_classes=1,vit_structure='SAM-H',freeze_encoder=True)
    model2 = CellViTSAM(input_classes=3,oputput_num_classes=1,vit_structure='SAM-H',freeze_encoder=True)
    # model3 = CellViTSAM(input_classes=3,oputput_num_classes=1,vit_structure='SAM-H',freeze_encoder=True)
    # model4 = CellViTSAM(input_classes=3,oputput_num_classes=1,vit_structure='SAM-H',freeze_encoder=True)
    
    state_dict1 = torch.load(model_dir1, map_location=torch.device('cpu'))['model_state_dict']
    state_dict2 = torch.load(model_dir2, map_location=torch.device('cpu'))['model_state_dict']
    # state_dict3 = torch.load(model_dir3, map_location=torch.device('cpu'))['model_state_dict']
    # state_dict4 = torch.load(model_dir4, map_location=torch.device('cpu'))['model_state_dict']
    
    model1.load_state_dict(state_dict1, strict=False)
    model2.load_state_dict(state_dict2, strict=False)
    # model3.load_state_dict(state_dict3, strict=False)
    # model4.load_state_dict(state_dict4, strict=False)
    
    print("Weights were successfully loaded!") #delete

    model1.to(device)
    model1.eval()
    
    model2.to(device)
    model2.eval()
    
    # model3.to(device)
    # model3.eval()
    
    # model4.to(device)
    # model4.eval()
    
    with torch.no_grad():
        cnt = 0
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (512, 512)
            sw_batch_size = 4
            b, c, h, w = val_images.size()
            #val_outputs = torch.zeros(b, 1, h, w).to(device)
            print(cnt)
            for scale in [0.75,1, 1.25]:
                val_images1 = F.interpolate(
                        val_images,
                        (int(scale * h), int(scale * w)),
                        mode="bilinear",
                        align_corners=True,
                    )
            
                val_outputs1 = sliding_window_inference(val_images1, roi_size, sw_batch_size, model1,padding_mode="reflect")
                
                val_outputs1 = F.interpolate(
                        val_outputs1, (h, w), mode="bilinear", align_corners=True
                    )
                
                
                val_images2 = F.interpolate(
                        val_images,
                        (int(scale * h), int(scale * w)),
                        mode="bilinear",
                        align_corners=True,
                    )
            
                val_outputs2 = sliding_window_inference(val_images2, roi_size, sw_batch_size, model2,padding_mode="reflect")
                
                val_outputs2 = F.interpolate(
                        val_outputs2, (h, w), mode="bilinear", align_corners=True
                    )
                
                # val_images3 = F.interpolate(
                #         val_images,
                #         (int(scale * h), int(scale * w)),
                #         mode="bilinear",
                #         align_corners=True,
                #     )
                # val_outputs3 = sliding_window_inference(val_images3, roi_size, sw_batch_size, model3,padding_mode="reflect")
                
                # val_outputs3 = F.interpolate(
                #         val_outputs3, (h, w), mode="bilinear", align_corners=True
                #     )
                
                
                
                # val_images4 = F.interpolate(
                #         val_images,
                #         (int(scale * h), int(scale * w)),
                #         mode="bilinear",
                #         align_corners=True,
                #     )
            
                # val_outputs4 = sliding_window_inference(val_images4, roi_size, sw_batch_size, model4,padding_mode="reflect")
                
                # val_outputs4 = F.interpolate(
                #         val_outputs4, (h, w), mode="bilinear", align_corners=True
                #     )
                
                # val_outputs = val_outputs1 + val_outputs2+ val_outputs3+ val_outputs4
                val_outputs = val_outputs1 + val_outputs2
                
            #val_outputs/=20
            val_outputs/=6    
                
                
                
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            # compute metric for current iteration
            # val_images = outputrans(val_images[0]).unsqueeze(0)
            # val_outputs = outputrans(val_outputs)
            dice_metric(y_pred=val_outputs, y=val_labels)
            # for val_output in val_outputs:
            #     saver(val_output)
            cnt = save_validate(val_images, val_labels, val_outputs, output_dir, images, cnt)
            #cnt+=1
        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())
        #print(model_dir) #delete
        # reset the status
        dice_metric.reset()


if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as tempdir:
    # input_dir = '/input/'
    # output_dir = '/output/'
    # wd = '/myhome/wd'
    # model_dir = '/model/'

    input_dir = '/ssd/cyj/KPIs24datanew/task1/validation/'
    model_dir1 = "/ssd/cyj/code/work_dir26_crop_huge1/sam_crop_huge/best_metric_model.pth"
    model_dir2 = "/ssd/cyj/code/work_dir26_crop_huge2/sam_crop_huge/best_metric_model.pth"
    # model_dir3 = "/ssd/cyj/code/work_dir26_crop_huge3/sam_crop_huge/best_metric_model.pth"
    # model_dir4 = "/ssd/cyj/code/work_dir26_crop_huge4/sam_crop_huge/best_metric_model.pth"
    
    output_dir = './task1_val_crop_huge/'
    #main(input_dir, model_dir1,model_dir2,model_dir3,model_dir4, output_dir)
    main(input_dir, model_dir1,model_dir2, output_dir)