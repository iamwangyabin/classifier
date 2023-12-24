##  Binary Classifier / Deepfake Detector

Mainly based on CNNDetection https://github.com/PeterWang512/CNNDetection

Just test some recent pretrained models, such as ViT, from timm for deepfake detection.




## (1) Setup

### Install packages
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Download model weights
- Run `bash weights/download_weights.sh`


## (2) Quick start

### Run on a single image

This command runs the model on a single image, and outputs the uncalibrated prediction.

```
# Model weights need to be downloaded.
python demo.py -f examples/real.png -m weights/blur_jpg_prob0.5.pth
```


## (3) Dataset

## (4) Train your models
We provide two example scripts to train our `Blur+JPEG(0.5)` and `Blur+JPEG(0.1)` models. We use `checkpoints/[model_name]/model_epoch_best.pth` as our final model.

change diffusiondb dataset

```
# Train Blur+JPEG(0.5)
CUDA_VISIBLE_DEVICES=0 python train.py --name vit_sd_blur_jpg_prob0.5 --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ~/workspace/datasets/aiart/output2/sd --arch vit

# Train Blur+JPEG(0.1)
CUDA_VISIBLE_DEVICES=3 python train.py --name vit_sd_blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ~/workspace/datasets/aiart/output2/sd --arch vit

CUDA_VISIBLE_DEVICES=2 python train.py --name resnet_sd_blur_jpg_prob0 --blur_prob 0 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ~/workspace/datasets/aiart/output2/sd --arch res50

CUDA_VISIBLE_DEVICES=3 python train.py --name vit_sd_blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ~/workspace/datasets/aiart/output2/sd --arch vit

CUDA_VISIBLE_DEVICES=0 python train.py --name vit_sd_blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /mnt/Data/yabin/datasets/aiart/base_session/sd --arch vit
```

## (5) Test your models


```
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot /mnt/Data/yabin/datasets/aiart/base_session/sd/val \
                                        --arch vit_base_patch16_224 \
                                        --model_path /home/yabin/CDDDet/checkpoints/vit_sd_blur_jpg_prob0.1/model_epoch_latest.pth \
                                        --sub_dirs dalle,imagen,mj,parti,sd,sdft \
                                        --results_dir ./sdbase
   
   
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot /mnt/Data/yabin/datasets/aiart/base_session/pd/val \
                                        --arch vit_base_patch16_224 \
                                        --model_path /home/yabin/CDDDet/checkpoints/vit_pd_blur_jpg_prob0.1/model_epoch_latest.pth \
                                        --sub_dirs dalle,imagen,mj,parti,sd,sdft \
                                        --results_dir ./pdbase
                                       
```






## (A) Acknowledgments

This repository borrows partially from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and the PyTorch [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories. 
