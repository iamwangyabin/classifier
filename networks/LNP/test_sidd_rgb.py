"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scipy.io as sio
from denoising_rgb import DenoiseNet
from dataloaders.data_rgb import get_validation_data
import util
import cv2
from skimage import img_as_ubyte
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/home/jwang/ybwork/data/DFBenchmark/Artifact',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/home/jwang/ybwork/data/deepfake_benchmark_LNP/Artifact',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='./sidd_rgb.pth',
                    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--noise_type', default=None,
                    type=str, help='e.g. jpg, blur, resize')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

util.mkdir(args.result_dir)

for dirpath, dirnames, filenames in os.walk(args.input_dir):
    print(dirpath)
    print(dirnames)

    image_paths = []
    for file in filenames:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            image_paths.append(os.path.join(dirpath, file))

    test_dataset = get_validation_data(image_paths, args.noise_type)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    print('storing into: ' + args.result_dir)
    model_restoration = DenoiseNet()
    util.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    with torch.no_grad():
        psnr_val_rgb = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_noisy = data_test[0].cuda()
            filenames = data_test[1]
            try:
                rgb_restored = model_restoration(rgb_noisy)
            except:
                print(filenames)
                continue
            rgb_restored = torch.clamp(rgb_restored, 0, 1)
            try:
                rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
                rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                for batch in range(len(rgb_noisy)):
                    denoised_img = img_as_ubyte(rgb_restored[batch])
                    imgsavepath = filenames[batch].replace(args.input_dir, args.result_dir)

                    imgsavepath = imgsavepath.replace('jpg', 'png').replace('JPEG', 'png').replace('jpeg', 'png')
                    try:
                        rootpath = imgsavepath.split('/')
                        rootpath.pop(-1)
                        rootpath = '/'.join(rootpath)
                        os.makedirs(rootpath)
                        print('root_path:' + rootpath)

                        cv2.imwrite(imgsavepath, denoised_img * 255)
                    except:
                        cv2.imwrite(imgsavepath, denoised_img * 255)
            except:
                print(filenames)
                continue
#
# from PIL import Image
# import os
# from tqdm.contrib.concurrent import process_map
#
# def resize_image(image_info):
#     input_path, output_path, quality, min_side_length = image_info
#     try:
#         with Image.open(input_path) as img:
#             if img.mode != 'RGB':
#                 img = img.convert('RGB')
#             width, height = img.size
#             if width < height:
#                 new_width = min_side_length
#                 new_height = int((new_width / width) * height)
#             else:
#                 new_height = min_side_length
#                 new_width = int((new_height / height) * width)
#             img_resized = img.resize((new_width, new_height), Image.LANCZOS)
#             img_resized.save(output_path, 'WEBP', quality=quality)
#     except Exception as e:
#         print(f"Error processing {input_path}: {e}")
#
#
# def process_folder(folder_path):
#     tasks = []
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith('.webp'):
#             input_path = os.path.join(folder_path, filename)
#             tasks.append((input_path, input_path, 90, 512))
#     process_map(resize_image, tasks, chunksize=1)
#
# for i in range(33, 67):
#     print(i)
#     process_folder("./group_{}".format(i))
\begin{tabular}{lrrrr}
\hline
 Method   &    &   Average_ACC &   Average_AP \\
\hline
 CNN01    &    &         50.42 &        61.65 \\
 CNN05    &    &         50.39 &        66.33 \\
 FreDect  &    &         60.03 &        62.42 \\
 Freqnet  &    &         45.05 &        44.58 \\
 Gram     &    &         49.74 &        39.95 \\
 NPRo     &    &         50.42 &        46.72 \\
 NPRr     &    &         50.78 &        43.57 \\
 Ojha     &    &         63.87 &        91.5  \\
 Our      &    &         73.24 &        80.26 \\
\hline
\end{tabular}


\begin{tabular}{lrrrr}
\hline
 Method   &    &   Average_F_ACC &   Average_R_ACC \\
\hline
 CNN01    &    &            0.07 &          100    \\
 CNN05    &    &            0    &          100    \\
 FreDect  &    &           43.77 &           76.03 \\
 Freqnet  &    &            7.15 &           82.36 \\
 Gram     &    &            0    &           98.71 \\
 NPRo     &    &            6.04 &           94.12 \\
 NPRr     &    &            7.02 &           93.86 \\
 Ojha     &    &           27.56 &           99.61 \\
 Our      &    &           62.14 &           84.17 \\
\hline
\end{tabular}






