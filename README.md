# KPIs2024



Solution of Team saltfish for KPIs2024: WSI-level Diseased Glomeruli  Instance Detection 

You can reproduce our method as follows step by step:


## Environments and Requirements


Our development environments:

| System                  | Ubuntu 22.04.4 LTS                        |
| ----------------------- | ----------------------------------------- |
| CPU                     | AMD EPYC 9554 64-Core Processor           |
| RAM                     | 16*32GB; 3200MT/s                         |
| GPU(number and type)    | 4 NVIDIA RTX A6000 48G                    |
| CUDA version            | 12.1                                      |
| Programming language    | Python 3.11.0                             |
| Deep learning framework | Pytorch (torch 2.1.2, torchvision 0.16.2) |
| Specific dependencies   | monai 1.3.0                               |

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

-  [MICCAI 2024 Kidney Pathology Image segmentation (KPIs) Challenge](https://sites.google.com/view/kpis2024/data).


## Training

To train the models, run this command :

  ```
  torchrun --nproc_per_node=<nproc_per_node> --data_path <data_patch> --work_dir <work_dir>  --model_name <model_name> --val_path <val_patch> train_final.py
  ```
You can also train with four cards using the default parameters£º
  ```
torchrun --nproc_per_node=4   train_final.py
  ```

Then we get models saved in 
```
./<work_dir> /<model_name>/ 
```
and logs saved in 
```
./<work_dir> /runs/
```
## Pre-trained Models
The publicly available pre-trained vit models can be download here:[Google Drive](https://drive.google.com/drive/folders/1UVwNHj9Y47j516SEUdtn1nlDau1kksDj)
Place the downloaded SAM's pre-trained ecoder weights in folder : ```model```
## Testing
You can download pre-trained weights here:[Google Drive](https://drive.google.com/drive/folders/1Zge3lp84ucAVZTOJVRGRY95jtPi_CQ5G)


To test the models, run this command :

  ```
  python unet_validation_slide.py --data_path <data_patch> --output_dir  <output_dir>  --model_dir1 <model_dir1> --model_dir2 <model_dir2>  --model_dir3 <model_dir3>  --model_dir4 <model_dir4> 
  ```

Docker  container link:[Docker Hub](https://hub.docker.com/repository/docker/cjjnihao/sf_wsl/general)
Docker reference: [kips2024 official code](https://github.com/hrlblab/KPIs2024)

docker run command:

```
docker pull cjjnihao/sf_wsl:slide3 #v3

# you need to specify the input directory
export data_dir=your_input_directory

# you need to specify the output directory
export output_dir=your_output_directory

mkdir $output_dir

docker run --rm -v $data_dir:/input_slide/:ro  -v $output_dir:/output_slide/ --gpus '"device=0"' -it cjjnihao/sf_wsl:slide3 #V3
```



## Results on test set
  | Method | Complete model |
  | ------ | -------------- |
  | dice   | 89.81          |
  | F1     | 91.33          |


## Acknowledgement

Many thanks to the organizing committee of this challenge, reference to [HoloHisto](https://arxiv.org/abs/2407.03307)
```
@misc{tang2024holohistoendtoendgigapixelwsi,
      title={HoloHisto: End-to-end Gigapixel WSI Segmentation with 4K Resolution Sequential Tokenization}, 
      author={Yucheng Tang and Yufan He and Vishwesh Nath and Pengfeig Guo and Ruining Deng and Tianyuan Yao and Quan Liu and Can Cui and Mengmeng Yin and Ziyue Xu and Holger Roth and Daguang Xu and Haichun Yang and Yuankai Huo},
      year={2024},
      eprint={2407.03307},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.03307}, 
}
```