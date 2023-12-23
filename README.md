##  Binary Classifier

**Dec 24th 2023 Update** first update


## How to use:

CUDA_VISIBLE_DEVICES=3 python train.py --name vit_sd_blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ~/workspace/datasets/aiart/output2/sd  --classes '' --arch vit

```
# Train Blur+JPEG(0.5)
CUDA_VISIBLE_DEVICES=0 python train.py --name vit_sd_blur_jpg_prob0.5 --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ~/workspace/datasets/aiart/output2/sd  --classes '' --arch vit


# Train Blur+JPEG(0.1)
CUDA_VISIBLE_DEVICES=3 python train.py --name vit_sd_blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ~/workspace/datasets/aiart/output2/sd  --classes '' --arch vit


CUDA_VISIBLE_DEVICES=2 python train.py --name resnet_sd_blur_jpg_prob0 --blur_prob 0 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ~/workspace/datasets/aiart/output2/sd  --classes '' --arch res50


CUDA_VISIBLE_DEVICES=0 python train.py --name vit_sd_blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /mnt/Data/yabin/datasets/aiart/base_session/sd  --classes '' --arch vit


```

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
```
# Train Blur+JPEG(0.5)
python train.py --name blur_jpg_prob0.5 --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse

# Train Blur+JPEG(0.1)
python train.py --name blur_jpg_prob0.1 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse
```

## (A) Acknowledgments

This repository borrows partially from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and the PyTorch [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories. 
