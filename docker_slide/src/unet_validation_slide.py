#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import cv2
import logging
import os
import sys
import tempfile
from glob import glob
from skimage import io, segmentation, morphology, measure, exposure,img_as_bool
import torch
from PIL import Image
from tifffile import imsave
from monai import config
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensityd, Resize, EnsureChannelFirstd
from matplotlib import cm
import matplotlib.pyplot as plt
import tifffile
import scipy.ndimage as ndi
import imageio
import numpy as np
from dataset_inf import MyDataset
from net_sam_infer import  CellViTSAM
import torch.nn.functional as F
from tqdm import tqdm

def calculate_contour_iou(contour1, contour2, image_shape):
    x1_min = contour1[:,:,0].min()
    x1_max = contour1[:,:,0].max()
    y1_min = contour1[:,:,1].min()
    y1_max = contour1[:,:,1].max()

    x2_min = contour2[:,:,0].min()
    x2_max = contour2[:,:,0].max()
    y2_min = contour2[:,:,1].min()
    y2_max = contour2[:,:,1].max()

    if x1_max < x2_min or x2_max < x1_min:
        return 0
    if y1_max < y2_min or y2_max < y1_min:
        return 0

    'crop'
    x_min = np.min([x1_min, x2_min]) - 10
    y_min = np.min([y1_min, y2_min]) - 10
    x_max = np.max([x1_max, x2_max]) + 10
    y_max = np.max([y1_max, y2_max]) + 10

    contour1[:,:,0] = contour1[:,:,0] - x_min
    contour1[:,:,1] = contour1[:,:,1] - y_min

    contour2[:,:,0] = contour2[:,:,0] - x_min
    contour2[:,:,1] = contour2[:,:,1] - y_min
    image_shape = (y_max - y_min, x_max - x_min)

    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 1, -1)
    cv2.drawContours(mask2, [contour2], -1, 1, -1)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def save_validate(val_images, output_dir, images, cnt):
    for i in range(val_images.shape[0]):
        folder_list = os.path.dirname(images[cnt+i]).split('/')
        save_folder = os.path.join(output_dir, folder_list[-2])

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        now_image = np.transpose(val_images[i], (2, 1, 0))
        
        name = os.path.basename(images[cnt+i])[0:-8]
        tifffile.imwrite(os.path.join(save_folder, '%s_mask.tiff' % (name)), now_image)


    cnt += val_images.shape[0]
    return cnt

def calculate_f1(precision, recall):
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Convert NaNs to zero if precision and recall are both zero
    return f1_scores


def dice_coefficient(mask1, mask2):
    # Convert masks to boolean arrays
    mask1 = np.asarray(mask1).astype(np.int8)
    mask2 = np.asarray(mask2).astype(np.int8)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)
    intersection_sum = np.sum(intersection)

    # Compute Dice coefficient
    if mask1_sum + mask2_sum == 0:  # To handle division by zero if both masks are empty
        return 1.0
    else:
        return 2 * intersection_sum / (mask1_sum + mask2_sum)


def calculate_metrics_ap50(pred_contours_list, gt_contours_list, image_shape, iou_thresholds=[0.5]):
    # Initialize lists to hold precision and recall values for each threshold
    precision_scores = []
    recall_scores = []

    for threshold in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0

        # Calculate matches for predictions
        for pred_contours in pred_contours_list:
            match_found = False
            for gt_contours in gt_contours_list:
                if calculate_contour_iou(pred_contours, gt_contours, image_shape) >= threshold:
                    tp += 1
                    match_found = True
                    break
            if not match_found:
                fp += 1

        # Calculate false negatives
        for gt_contours in gt_contours_list:
            if not any(calculate_contour_iou(pred_contours, gt_contours, image_shape) >= threshold for pred_contours in pred_contours_list):
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Compute F1 scores
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_scores, recall_scores)]
    return precision_scores, recall_scores, f1_scores


def sodelete(wsi, min_size):
    """
    Remove objects smaller than min_size from binary segmentation image.

    Args:
    img (numpy.ndarray): Binary image where objects are 255 and background is 0.
    min_size (int): Minimum size of the object to keep.

    Returns:
    numpy.ndarray: Image with small objects removed.
    """
    # Find all connected components (using 8-connectivity, as default)
    _, binary = cv2.threshold(wsi* 255, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8, cv2.CV_32S)

    # Create an output image that will store the filtered objects
    # output = np.zeros_like(wsi, dtype=np.uint8)
    output = np.zeros_like(wsi)

    # Loop through all found components
    for i in range(1, num_labels):  # start from 1 to ignore the background
        size = stats[i, cv2.CC_STAT_AREA]

        # If the size of the component is larger than the threshold, copy it to output
        if size >= min_size:
            output[labels == i] = 1.

    return output


def calculate_ap50(precisions, recalls):
    # Ensure that the arrays are sorted by recall
    sort_order = np.argsort(recalls)
    precisions = np.array(precisions)[sort_order]
    recalls = np.array(recalls)[sort_order]

    # Pad precisions array to include the precision at recall zero
    precisions = np.concatenate(([0], precisions))
    recalls = np.concatenate(([0], recalls))

    # Calculate the differences in recall to use as weights in weighted average
    recall_diff = np.diff(recalls)

    # Compute the average precision
    ap50 = np.sum(precisions[:-1] * recall_diff)
    return ap50


def main(datadir, model_dir1,model_dir2,model_dir3,model_dir4,output_dir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_transforms = Compose([EnsureChannelFirstd(keys=["input"],channel_dim='2'),
                                EnsureChannelFirstd(keys=["output"],channel_dim='no_channel'),
                                ScaleIntensityd(keys=["input","output"])])
    
    
    val_ds= MyDataset(datadir,transform=train_transforms)
    print(len(val_ds))

    print('total image: %d' % (len(val_ds)))

    # define transforms for image and segmentation
    # 2048 ->window ��2W->window
    #imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize(spatial_size=(512, 512), mode='nearest'), ScaleIntensity()])
    # imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    # segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    #outputrans = Compose([Resize(spatial_size=(2048, 2048), mode='nearest')])
    # val_ds = ArrayDataset(images, imtrans, segs, segtrans)
    # sliding window inference for one image at every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    
    model1 = CellViTSAM(input_classes=3,oputput_num_classes=1,vit_structure='SAM-H',freeze_encoder=True)
    model2 = CellViTSAM(input_classes=3,oputput_num_classes=1,vit_structure='SAM-H',freeze_encoder=True)
    model3 = CellViTSAM(input_classes=3,oputput_num_classes=1,vit_structure='SAM-H',freeze_encoder=True)
    #model4 = CellViTSAM(input_classes=3,oputput_num_classes=1,vit_structure='SAM-H',freeze_encoder=True)
    
    state_dict1 = torch.load(model_dir1, map_location=torch.device('cpu'))['model_state_dict']
    state_dict2 = torch.load(model_dir2, map_location=torch.device('cpu'))['model_state_dict']
    state_dict3 = torch.load(model_dir3, map_location=torch.device('cpu'))['model_state_dict']
    #state_dict4 = torch.load(model_dir4, map_location=torch.device('cpu'))['model_state_dict']
    
    model1.load_state_dict(state_dict1, strict=False)
    model2.load_state_dict(state_dict2, strict=False)
    model3.load_state_dict(state_dict3, strict=False)
    #model4.load_state_dict(state_dict4, strict=False)

    
    model1.to(device)
    model2.to(device)
    model3.to(device)
    #model4.to(device)
    
    model1.eval()
    model2.eval()
    model3.eval()
    #model4.eval()
    print("Weights were successfully loaded!", flush=True)#delete
    

    with torch.no_grad():
        cnt = 0
        
        wsi_F1_50 = []
        wsi_AP50 = []
        wsi_dice = []
        print("Please be patient as the image is too large.The progress bar appears, indicating that the program is normal", flush=True)
        progress_bar = tqdm(total=len(val_loader), desc="Inference progress", ncols=80)
        for val_data in val_loader:
            val_images, val_labels,imgaes_path= val_data[0].to(device),val_data[1].to(device), val_data[2]
            # define sliding window size and batch size for windows inference
            roi_size = (512, 512)
            sw_batch_size = 1
        

            val_outputs1 = sliding_window_inference(val_images, roi_size, sw_batch_size, model1,padding_mode="reflect")
            

            val_outputs2 = sliding_window_inference(val_images, roi_size, sw_batch_size, model2,padding_mode="reflect")
        

            val_outputs3 = sliding_window_inference(val_images, roi_size, sw_batch_size, model3,padding_mode="reflect")
            

        
            #val_outputs4 = sliding_window_inference(val_images, roi_size, sw_batch_size, model4,padding_mode="reflect")
            
            
            val_outputs = val_outputs1 + val_outputs2 + val_outputs3
   
                
            val_outputs/=3
            
            
            
            #val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            
            
            
            
            
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels) #list
            # compute metric for current iteration
            # val_images = outputrans(val_images[0]).unsqueeze(0)
            # val_outputs = outputrans(val_outputs)


            #sm = 20000
            val_outputs = torch.stack(val_outputs)
            
            
            val_outputs=val_outputs.cpu().numpy()
            wsi_prediction_sm = val_outputs.astype(np.float32)
            #print("val_outputs0:",np.unique(wsi_prediction_sm, return_counts=True))
            # wsi_prediction_sm[wsi_prediction_sm <= 1] = 0
            # wsi_prediction_sm[wsi_prediction_sm != 0] = 1
            
            #wsi_prediction_sm = sodelete(wsi_prediction_sm , sm)
            # print("val_outputs1:",np.unique(wsi_prediction_sm, return_counts=True))
            
            wsi_prediction_sm= img_as_bool(wsi_prediction_sm)
            #print("val_outputs2:",np.unique(wsi_prediction_sm, return_counts=True))
            wsi_prediction_sm = morphology.remove_small_objects(wsi_prediction_sm, min_size=10000)
            #print("val_outputs delete 0:",np.unique(wsi_prediction_sm, return_counts=True))
            
            wsi_prediction_sm = morphology.remove_small_holes(wsi_prediction_sm, area_threshold=10000)
            wsi_prediction_sm= wsi_prediction_sm.astype(float)
            #print("val_outputs hole:",np.unique(wsi_prediction_sm, return_counts=True))

            #plt.imsave(preds_root, wsi_prediction_sm, cmap=cm.gray)
            #imsave('output.tif', array)
            
            val_labels= torch.stack(val_labels)
            now_img_shape=val_outputs.shape
            wsi_mask=val_labels.cpu().numpy()
            
            
            #save

                
            basename=os.path.basename(imgaes_path[0])
            new_basename = basename.replace("wsi", "mask")
            case=imgaes_path[0].split('/')[-3]
            output=os.path.join(output_dir, case)
            if not os.path.exists(output):

                os.makedirs(output)
                
           # print("wsi_prediction_sm.shape 1",wsi_prediction_sm.shape)
            
            # 'f1'
            
            
            wsi_prediction_sm = np.squeeze(wsi_prediction_sm, axis=0)
            wsi_prediction_sm = np.squeeze(wsi_prediction_sm, axis=0)
            #print("wsi_prediction_sm.shape 2",wsi_prediction_sm.shape)
            
            plt.imsave(os.path.join(output,new_basename), wsi_prediction_sm, cmap=cm.gray)
            
            
            ret, binary = cv2.threshold(wsi_prediction_sm, 0, 255, cv2.THRESH_BINARY_INV)
            _,preds_contours, _ = cv2.findContours(binary.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)# opencv 3.x
            #preds_contours, _ = cv2.findContours(binary.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)# opencv 4.x
            wsi_mask = np.squeeze(wsi_mask, axis=0)
            wsi_mask = np.squeeze(wsi_mask, axis=0)
            
            ret, binary = cv2.threshold(wsi_mask, 0, 255, cv2.THRESH_BINARY_INV)
            _,masks_contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)# opencv 3.x
            #masks_contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)# opencv 4.x
            precision_scores, recall_scores, f1_scores_50 = calculate_metrics_ap50(preds_contours[1:], masks_contours[1:],(now_img_shape[-2], now_img_shape[-1]))
            ap50 = precision_scores


            wsi_F1_50.append((f1_scores_50[0]))
            wsi_AP50.append((ap50[0]))
            wsi_dice.append((dice_coefficient(wsi_prediction_sm, wsi_mask)))

            progress_bar.update(1)
            print("\n", flush=True)
            print(new_basename, flush=True)
            print("slide level F1 metric:", (f1_scores_50[0]), flush=True)
            print("slide level AP(50) metric:", (ap50[0]), flush=True)
            print("slide level Dice metric:", (dice_coefficient(wsi_prediction_sm, wsi_mask)), flush=True)

            print("\n", flush=True)
        progress_bar.close()
        print("\n", flush=True)
        print("Average:", flush=True)
        print("slide level F1 metric:", np.mean(wsi_F1_50), flush=True)
        print("slide level AP(50) metric:", np.mean(wsi_AP50), flush=True)
        print("slide level Dice metric:", np.mean(wsi_dice), flush=True)
        
        #print(model_dir)


if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as datadir:
    # data_dir = '/input_slide/'
    # output_dir = '/output_slide/'
    # patch_data_dir =  '/input_patch/'
    # model_dir = '/model/'
    # patch_output_dir = '/output_patch/'
    # wd = '/myhome/wd'

    # data_dir = '/ssd/cyj/KPIs24datanew/task2/val/'
    # output_dir = './task2_val/output/'
    # patch_data_dir = './task2_val/testing_data_wsi_patch_20X/'
    # model_dir = "/ssd/cyj/code/work_dir18/samh_unet_final/best_metric_model.pth"
    # patch_output_dir = './task2_val/validation_slide_20X_patchoutput/'
    
    
    ##############################

    # data_dir = '/input_slide/'
    # output_dir = '/output_slide/'
    # wd = '/myhome/wd'
    
    model_dir1 = "./src/model/model1.pth"
    model_dir2 = "./src/model/model2.pth"
    model_dir3 = "./src/model/model3.pth"
    model_dir4 = "./src/model/model4.pth"
    
    
    ###########################cs
    #os.chdir("/ssd/cyj/code/")
    data_dir = '/ssd/cyj/KPIs24datanew/task2/val/'
    output_dir = '/ssd/cyj/KPIs2024/task2_val/'
    # model_dir1 = "/ssd/cyj/code/work_dir26_crop_huge1/sam_crop_huge/best_metric_model.pth"
    # model_dir2 = "/ssd/cyj/code/work_dir26_crop_huge2/sam_crop_huge/best_metric_model.pth"
    # model_dir3 = "/ssd/cyj/code/work_dir26_crop_huge3/sam_crop_huge/best_metric_model.pth"
    # model_dir4 = "/ssd/cyj/code/work_dir26_crop_huge4/sam_crop_huge/best_metric_model.pth"
    
    
####################################



    # data_dir = '/Data/KPIs/data_val_slide'
    # output_dir = '/Data/KPIs/validation_slide_20X'
    # patch_data_dir = '/Data/KPIs/testing_data_wsi_patch_20X'
    # model_dir = '/Data/KPIs/checkpoint'
    # patch_output_dir = '/Data/KPIs/validation_slide_20X_patchoutput'

    main(data_dir, model_dir1, model_dir2 , model_dir3 , model_dir4, output_dir)
