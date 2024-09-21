#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""
import pdb
import warnings
import time
from torch.nn.functional import sigmoid
from tqdm import tqdm
warnings.filterwarnings("ignore")
# from torchmetrics import IoU
from torch.optim.lr_scheduler import LambdaLR
from monai.inferers import sliding_window_inference
import torch.nn.functional as F
from monai.data import decollate_batch, PILReader
from skimage import io, segmentation, morphology, measure, exposure
from torchvision.utils import save_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage import feature
from pickle import FALSE
import sys
import argparse
import os
import cv2
join = os.path.join
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import decollate_batch, PILReader,NumpyReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    EnsureTyped,

)
from monai.transforms import Activations, AsDiscrete, Compose
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import sys

from utils import Dice_Loss
from utils.miou import compute_miou, fast_hist, per_class_iu, compute_mIoU
# from torch.autograd import Variable

from net_sam import SAM_Model , CellViTSAM
os.environ['MASTER_ADDR'] = 'localhost' 
# os.environ['MASTER_PORT'] = '30020'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

print("Successfully imported all requirements!")
from pathlib import Path
import re
import numpy as np


def main():
    parser = argparse.ArgumentParser("KPIs2024")
    # Dataset parameters
    
    parser.add_argument(
        "--data_path",
        default="/ssd/cyj/KPIs24datanew/task1/train",
        type=str,
        help="training  data path",
    )
    
    parser.add_argument(
        "--val_path",
        default="/ssd/cyj/KPIs24datanew/task1/validation",
        type=str,
        help="training val path; subfolders",
    )
    
    parser.add_argument(
        "--work_dir", default="./work_dir29_crop_huge", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=2024, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=24, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", default="sam_crop_huge", help="select mode: res50_unetr, res50wt_unetr,"
    )
    parser.add_argument("--num_class", default=1, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=512, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=24, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=2e-4, help="learning rate") #ecoder -4
    parser.add_argument("--warmup", type=bool, default=False, help="learning rate")
    parser.add_argument("--warmup_period", type=int, default=50, help="learning rate")
    parser.add_argument('--lr_exp', type=float, default=0.85, help='The learning rate decay expotential')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    ### gpu id
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", init_method='env://', rank=local_rank, world_size=4)
    monai.config.print_config()
    np.random.seed(args.seed)

    model_path = join(args.work_dir, args.model_name)
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )

    data=[]
    root_dir= args.data_path
    #data
    for label_folder in os.listdir(root_dir):
    #56NX、DN
        for subfolder in os.listdir(os.path.join(root_dir, label_folder)):
            #12_116
            img_folder = os.path.join(root_dir, label_folder, subfolder, 'img')
            mask_folder = os.path.join(root_dir, label_folder, subfolder, 'mask')

            # all jpg files in img_folder
            for filename in os.listdir(img_folder):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(img_folder, filename)
                    mask_path = os.path.join(mask_folder, filename[:-8] + '_mask.jpg')
                    data.append({"name":filename,"img": img_path, "label": mask_path})
    

    img_num = len(data)
    val_frac = 0.0 # 0.1 # extra
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    np.random.seed(args.seed+local_rank)
    
    train_files = [
            data[i]
        for i in train_indices
    ]


    data_val=[]
    val_root_dir= args.val_path
    #data
    for label_folder in os.listdir(val_root_dir):
    #56NX、DN
        for subfolder in os.listdir(os.path.join(val_root_dir, label_folder)):
            #12_116
            img_folder = os.path.join(val_root_dir, label_folder, subfolder, 'img')
            mask_folder = os.path.join(val_root_dir, label_folder, subfolder, 'mask')
            for filename in os.listdir(img_folder):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(img_folder, filename)
                    mask_path = os.path.join(mask_folder, filename[:-8] + '_mask.jpg')
                    data_val.append({"name":filename,"img": img_path, "label": mask_path})


    val_indices=np.arange(len(data_val))


    val_files = [
            data_val[i]
        for i in val_indices
    ]
    
    
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    
    print("data_ok!")
    
    #%% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img","label"], reader=PILReader, dtype=np.float32
            ),  
            
            EnsureChannelFirstd(keys=["img","label"], allow_missing_keys=True), 
            
            ScaleIntensityd(
                keys=["img","label"], allow_missing_keys=True,
            ),  
            RandZoomd(
                keys=["img","label"],
                prob=1,
                min_zoom=0.25,
                max_zoom=2,
                mode=["area","nearest"],
                keep_size=False,
            ),
            
            RandSpatialCropd(
                keys=["img", "label"], roi_size=(args.input_size,args.input_size), random_size=False
            ),
            
            RandAxisFlipd(keys=["img","label"], prob=0.5),
            RandRotate90d(keys=["img","label"], prob=0.5, spatial_axes=[0, 1]),
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
            EnsureTyped(keys=["img","label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img","label"], reader=PILReader, dtype=np.float32),
    
            EnsureChannelFirstd(keys=["img","label"], allow_missing_keys=True), 
            
            ScaleIntensityd(
                keys=["img","label"], allow_missing_keys=True,
            ), 
            EnsureTyped(keys=["img","label"]),
              
        ]
    )
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    if dist.get_rank() == 0:
        check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        check_loader = DataLoader(check_ds, batch_size=12, num_workers=24)
        check_data = monai.utils.misc.first(check_loader)
        print(
            "sanity check:",
            "img:",check_data["img"].shape, 
            "label:",check_data["label"].shape, 
            "img max:",torch.max(check_data["img"]),
            "img min:",torch.min(check_data["img"]),
            "label max:",torch.max(check_data["label"]),
            "label min:",torch.min(check_data["label"])
        )
        
    args.batch_size = args.batch_size // torch.cuda.device_count()
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler= train_sampler,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=24)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    
 
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CellViTSAM(model_path='./model/sam_vit_h.pth',input_classes=3,continue_train=False,oputput_num_classes=args.num_class,vit_structure='SAM-H',freeze_encoder=True)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device) 
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=False)
    
    # loss_function = monai.losses.DiceFocalLoss(softmax = False)
    loss_function_bce = nn.BCEWithLogitsLoss()
    #loss_function_dic = monai.losses.DiceCELoss() 
    #loss_function_dic =monai.losses.DiceCELoss(to_onehot_y=False, softmax=False)

    loss_function_dic  = Dice_Loss.BinaryDiceLoss()
    

    base_lr = args.initial_lr
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    # start a typical PyTorch training
    max_epochs = args.max_epochs

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    # dsc_scores = list()
    
    val_interval = args.val_interval
    epoch_tolerance = args.epoch_tolerance
    if dist.get_rank() == 0:
        
        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        log_dir = './{}/runs/{}+{}'.format(args.work_dir,current_time, args.model_name)

        writer = SummaryWriter(log_dir=log_dir)

    # writer = SummaryWriter(model_path)
    iter_num = 0
    max_iterations = args.max_epochs * len(train_loader)
    for epoch in range(1, max_epochs+1):
        # train_sampler.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        loss_dic = 0
        loss_bce = 0
        if dist.get_rank() == 0:####
            pbar = tqdm(total=len(train_loader))####
            
        for step, batch_data in enumerate(train_loader, 1):
            name,img,mask = batch_data['name'], batch_data["img"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()
        
            outputs= model(img) 
            

            loss1 = loss_function_bce(outputs,mask) #need sigmoid
            
            outputs= torch.sigmoid(outputs) 
            
            loss2 = loss_function_dic(outputs,mask) #binarydiceloss need sigmoid
            
            loss =loss1+loss2    
             
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_  
                    
            iter_num = iter_num + 1 

            epoch_loss += loss.item()
            # print("epoch_loss",epoch_loss.device)
            loss_dic += loss2.item()
            loss_bce += loss1.item()
            
            epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            if dist.get_rank() == 0:
                
                #pbar.set_postfix(postfix)######
            
                writer.add_scalar("bce_loss", loss1.item(), epoch_len * epoch + step)
                writer.add_scalar("dice_loss", loss2.item(), epoch_len * epoch + step) 
                
                pbar.update(1)#####
                
        epoch_loss /= step
        loss_dic/= step
        loss_bce /= step
        epoch_loss_values.append(epoch_loss)
        if dist.get_rank() == 0:
            pbar.close()#######
            print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
            print(f"epoch {epoch} average loss bce: {loss_bce:.4f}")
            print(f"epoch {epoch} average loss dic: {loss_dic:.4f}")
            print("lr:",optimizer.param_groups[0]['lr'])  
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict(),
                # "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss_values,
            }
            
            if epoch % 5 ==0:
                
                torch.save(checkpoint, join(model_path, f"eopch_{epoch}_model.pth"))
                
                model.eval()
                
                val_epoch_loss = 0
                val_loss_dic = 0
                val_loss_bce = 0
                metric=0
                
                with torch.no_grad():
                        pbar = tqdm(total=len(val_loader))####
                        dsc_scores = list()
                        for step, batch_data in enumerate(val_loader, 1):
                            
                            if step==32:
                                break
                            name,val_images,val_labels = batch_data['name'], batch_data["img"].to(device), batch_data["label"].to(device)

                            roi_size = (512, 512)
                            
                            sw_batch_size = 4
                            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                            
                            val_loss1 = loss_function_bce(val_outputs,val_labels) 
                            val_outputs2= torch.sigmoid(val_outputs)
                            val_loss2 = loss_function_dic(val_outputs2,val_labels) 
                            
                            
                            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                            # compute metric for current iteration
                            dsc =dice_metric(y_pred=val_outputs, y=val_labels)
                            
                            not_nans = torch.isfinite(dsc)
                            
                            #pdb.set_trace()
                            
                            dsc_scores.append(dsc[not_nans].mean().item())
                            
                            val_loss =val_loss1+val_loss2
                            
                            val_epoch_loss += val_loss
                            # print("epoch_loss",epoch_loss.device)
                            val_loss_dic += val_loss2
                            val_loss_bce += val_loss1
                            
                            pbar.update(1)
                        dsc_scores = torch.tensor(dsc_scores)
                        not_nans = torch.isfinite(dsc_scores)
                        metric = sum(dsc_scores[not_nans]) / len(dsc_scores[not_nans])
                        # metric/=step 
                        val_loss_bce /= step
                        val_loss_dic/= step
                        val_epoch_loss /= step
                        
                        metric_values.append(metric)
                        
                        if metric > best_metric:
                            best_metric=metric
                            best_metric_epoch=epoch
                            torch.save(checkpoint, join(model_path,"best_metric_model.pth"))
                            
                        print(f"epoch {epoch} val_epoch_loss: {val_epoch_loss:.4f}")
                        print(f"epoch {epoch} val_loss_bce: {val_loss_bce:.4f}")
                        print(f"epoch {epoch} val_loss_dic: {val_loss_dic:.4f}")
                        print(f"epoch {epoch} val_dsc_loss: {metric:.4f}")
                        print(f"best_metric: {best_metric:.4f} best_metric_epoch: {best_metric_epoch}")
                        
                        writer.add_scalar("val_loss_bce", val_loss_bce, epoch )
                        writer.add_scalar("val_loss_dic", val_loss_dic, epoch ) 
                        writer.add_scalar("val_dsc_loss", metric, epoch ) 
                        
                        pbar.close()
                        
            if epoch == args.max_epochs:
                torch.save(checkpoint, join(model_path, "final_model.pth"))
    if dist.get_rank() == 0:    
        writer.close()

if __name__ == "__main__":
    main()
