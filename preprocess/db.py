import json
from collections import Counter
import dateutil.parser
import yaml
from transformers import pipeline
from torchvision import transforms
import os
from PIL import ImageFile, Image
from tqdm import tqdm
import csv
import torch
from torch.utils.data import Dataset, DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

src_images_dir = "/data/jwang/db2024extracted"
dest_dir = "/home/jwang/ybwork/data/dbv3"


data = []
with open("/data/jwang/db2024/metadata/posts.json", "r") as file:
    for line in file:
        try:
            json_obj = json.loads(line)
            data.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

def generate_tags(data_item):
    def process_tags(tag_str):
        processed_tags = []
        for tag_name in tag_str.split(" "):
            if len(tag_name) > 3:
                tag_name = tag_name.replace("_", " ")
            processed_tags.append(tag_name)
        return ", ".join(processed_tags)
    created_at = data_item.get("media_asset", {}).get("created_at", "")
    try:
        parsed_date = dateutil.parser.isoparse(created_at)
        year = parsed_date.year
        if 2005 <= year <= 2010:
            year_tag = "oldest"
        elif 2011 <= year <= 2014:
            year_tag = "early"
        elif 2015 <= year <= 2018:
            year_tag = "mid"
        elif 2019 <= year <= 2021:
            year_tag = "late"
        elif 2022 <= year <= 2023:
            year_tag = "newest"
        else:
            year_tag = "unknown"
    except (ValueError, AttributeError):
        print("Invalid or missing created_at date.")
        year_tag = "unknown"
    score = data_item.get("score")
    tags_general = process_tags(data_item.get("tag_string_general", ""))
    tags_character = process_tags(data_item.get("tag_string_character", ""))
    if tags_character == "original":
        tags_character = ""
    else:
        tags_character = tags_character
    tags_artist = data_item.get("tag_string_artist", "")
    if tags_artist == "":
        tags_artist = ""
    else:
        # tags_artist = process_tags(data_item.get("tag_string_artist", ""))
        tags_artist = tags_artist
    tags_meta = process_tags(data_item.get("tag_string_meta", ""))
    quality_tag = ""
    if score > 150:
        quality_tag = "masterpiece, "
    elif 100 <= score <= 150:
        quality_tag = "best quality"
    elif 75 <= score < 100:
        quality_tag = "high quality"
    elif 25 <= score < 75:
        quality_tag = "medium quality"
    elif 0 <= score < 25:
        quality_tag = "normal quality"
    elif -5 <= score < 0:
        quality_tag = "low quality"
    elif score < -5:
        quality_tag = "worst quality"
    tags_general_list = tags_general.split(', ')
    special_tags = [
        "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple girls",
        "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple boys", "male focus"
    ]
    found_special_tags = [tag for tag in tags_general_list if tag in special_tags]
    for tag in found_special_tags:
        tags_general_list.remove(tag)
    first_general_tag = ', '.join(found_special_tags)
    rest_general_tags = ', '.join(tags_general_list)
    # tags_separator = "|||"
    pre_separator_tags = []
    post_separator_tags = []
    if tags_character:
        pre_separator_tags.append(tags_character)
    if tags_artist != "":
        pre_separator_tags.append(tags_artist)
    if first_general_tag:
        pre_separator_tags.append(first_general_tag)
    if rest_general_tags:
        post_separator_tags.append(rest_general_tags)
    if year_tag:
        post_separator_tags.append(year_tag)
    if tags_meta:
        post_separator_tags.append(tags_meta)
    if quality_tag:
        post_separator_tags.append(quality_tag)
    pre_separator_str = ', '.join(pre_separator_tags)
    post_separator_str = ', '.join(post_separator_tags)
    caption = f"{pre_separator_str}, {post_separator_str}"
    return caption, tags_artist



def build_file_index(src_images_dir):
    file_index = {}
    for root, _, files in os.walk(src_images_dir):
        for file in files:
            file_name, file_ext = os.path.splitext(file)
            file_index[file_name] = os.path.join(root, file)
    return file_index

file_index = build_file_index(src_images_dir)


def get_processed_file_aesthetic(csv_file):
    file_aesthetic = {}
    try:
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if row:
                    file_aesthetic[row[0]] = row[2]
    except FileNotFoundError:
        pass
    return file_aesthetic

# 以美学指标生成图片
file_aesthetic = get_processed_file_aesthetic('aesthetic_scores.csv')

# 生成图片 2.5m
def process_image(i, file_index=file_index, dest_dir=dest_dir):
    try:
        if str(i['id']) in file_aesthetic and float(file_aesthetic[str(i['id'])]) > 0.9 and float(file_aesthetic[str(i['id'])]) < 0.93:
            if i['image_width'] >= 768 and i['image_height'] >= 768:
                caption, tags_artist = generate_tags(i)
                i['tag_string_artist'] = ""
                file_path = file_index.get(str(i['id']))
                with Image.open(file_path) as img:
                    min_size = 1024
                    scale = min_size / min(img.size)
                    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                    img = img.resize(new_size, Image.LANCZOS)
                    img.save(os.path.join(dest_dir, 's1', str(i['id']) + ".jpg"), 'JPEG')
                    with open(os.path.join(dest_dir, 's1', str(i['id']) + '.txt'), 'w', encoding='utf-8') as file:
                        file.write(caption)
        elif str(i['id']) in file_aesthetic and float(file_aesthetic[str(i['id'])]) >= 0.93 and float(file_aesthetic[str(i['id'])]) < 0.95:
            if i['image_width'] >= 768 and i['image_height'] >= 768:
                caption, tags_artist = generate_tags(i)
                i['tag_string_artist'] = ""
                file_path = file_index.get(str(i['id']))
                with Image.open(file_path) as img:
                    min_size = 1024
                    scale = min_size / min(img.size)
                    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                    img = img.resize(new_size, Image.LANCZOS)
                    img.save(os.path.join(dest_dir, 's2', str(i['id']) + ".jpg"), 'JPEG')
                    with open(os.path.join(dest_dir, 's2', str(i['id']) + '.txt'), 'w', encoding='utf-8') as file:
                        file.write(caption)
        elif str(i['id']) in file_aesthetic and float(file_aesthetic[str(i['id'])]) >= 0.95:
            if i['image_width'] >= 768 and i['image_height'] >= 768:
                caption, tags_artist = generate_tags(i)
                i['tag_string_artist'] = ""
                file_path = file_index.get(str(i['id']))
                with Image.open(file_path) as img:
                    min_size = 1024
                    scale = min_size / min(img.size)
                    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                    img = img.resize(new_size, Image.LANCZOS)
                    img.save(os.path.join(dest_dir, 's3', str(i['id']) + ".jpg"), 'JPEG')
                    with open(os.path.join(dest_dir, 's3', str(i['id']) + '.txt'), 'w', encoding='utf-8') as file:
                        file.write(caption)
    except Exception as e:
        print(f"Error processing image {i['id']}: {e}")



os.makedirs(dest_dir, exist_ok=True)
os.makedirs(os.path.join(dest_dir, 's2'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 's3'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 's1'), exist_ok=True)

# 这么并行处理比较简单而且快速，pytorch会解决一切多进程问题
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, data, file_index=file_index, dest_dir=dest_dir):
        self.data = data
        self.file_index = file_index
        self.dest_dir = dest_dir
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        a = process_image(self.data[idx], self.file_index, self.dest_dir)
        return idx

dataset = SimpleDataset(data, file_index, dest_dir)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

for batch in tqdm(dataloader):
    pass


def get_aesthetic():
    from datasets import Dataset, Image
    pipe_aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic", device="cuda:1")
    image_folder = '/data/jwang/db2024extracted'

    def get_processed_file_ids(csv_file):
        processed_ids = set()
        try:
            with open(csv_file, mode='r', newline='') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    if row:
                        processed_ids.add(row[0])
        except FileNotFoundError:
            pass
        return processed_ids

    processed_file_ids = get_processed_file_ids('aesthetic_scores.csv')
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg', 'webp')) and os.path.basename(img).split('.')[0] not in processed_file_ids]
    # dataset = Dataset.from_dict({"image": image_paths}).cast_column("image", Image())
    keys = [os.path.basename(img).split('.')[0] for img in image_paths]
    dataset = Dataset.from_dict({"image": image_paths, "key": keys}).cast_column("image", Image())
    transform_list = [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    dataset.set_format("torch")

    def custom_trans(examples):
        images = []
        keys = []
        for image, key in zip(examples["image"], examples["key"]):
            try:
                images.append(trans(image.convert("RGB")))
                keys.append(key)
            except:
                print(f"Error processing image {image}: {e}")
        examples['image'] = images
        examples['key'] = keys
        return examples

    dataset.set_transform(custom_trans)

    asdata_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=32)

    model = pipe_aesthetic.model
    model.eval()

    # all_aesthetic = {}
    # for batch in tqdm(asdata_loader):
    #     images = batch['image']
    #     file_ids = batch['key']
    #     images = images.to(model.device)
    #     with torch.no_grad():
    #         outputs = model(images)
    #     for idx, file_id in enumerate(file_ids):
    #         all_aesthetic[file_id] = outputs['logits'][idx].tolist()

    # >>> pipe_aesthetic.postprocess(outputs)
    # [{'score': 0.7624236941337585, 'label': 'aesthetic'}, {'score': 0.23757633566856384, 'label': 'not_aesthetic'}]
    # >>> outputs['logits'][0]
    # tensor([-1.4494, -0.2834], device='cuda:1')
    # >>> outputs['logits'][0].softmax(-1)[0]
    # tensor(0.2376, device='cuda:1')
    # >>> outputs['logits'][0].softmax(-1)
    # tensor([0.2376, 0.7624], device='cuda:1')

    # with open('aesthetic_scores.csv', mode='w', newline='') as file:
    #     csv_writer = csv.writer(file)
    #     csv_writer.writerow(['file_id', 'aesthetic_score_1', 'aesthetic_score_2'])
    #     for batch in tqdm(asdata_loader):
    #         images = batch['image']
    #         file_ids = batch['key']
    #         images = images.to(model.device)
    #         with torch.no_grad():
    #             outputs = model(images)
    #         for idx, file_id in enumerate(file_ids):
    #             aesthetic_scores = outputs['logits'][idx].softmax(-1).cpu().tolist()
    #             csv_writer.writerow([file_id] + aesthetic_scores)
    # resume
    with open('aesthetic_scores.csv', mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        if not processed_file_ids:
            csv_writer.writerow(['file_id', 'aesthetic_score_1', 'aesthetic_score_2'])
        for batch in tqdm(asdata_loader):
            images = batch['image']
            file_ids = batch['key']
            images = images.to(model.device)
            with torch.no_grad():
                outputs = model(images)
            for idx, file_id in enumerate(file_ids):
                aesthetic_scores = outputs['logits'][idx].softmax(-1).cpu().tolist()
                csv_writer.writerow([file_id] + aesthetic_scores)

# 运行
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision='fp16' -- -m hcpdiff.train_ac_single --cfg cfgs/train/ft_playground.yaml


import os
import torch
import h5py

# 定义源目录和目标文件
directory = "./data/dbv2_cache"
target_h5_file = './data/dbv2_cache.h5'

# 创建一个新的HDF5文件
with h5py.File(target_h5_file, 'w') as h5file:
    for filename in os.listdir(directory):
        if filename.endswith(".pth"):
            file_path = os.path.join(directory, filename)
            data = torch.load(file_path, map_location=torch.device('cpu'))
            names, ls, mask, crop_info = data
            for img_name, feature, m, c in zip(names, ls, mask, crop_info):
                if str(img_name) in h5file:
                    print(f"Dataset '{str(img_name)}' already exists.")
                else:
                    feature_np = feature.numpy()
                    h5file.create_dataset(str(img_name), data=feature_np)


# accelerate launch --num_processes=4 --num_machines=1 --mixed_precision='bf16' -m hcpdiff.train_ac --cfg cfgs/train/ft_playground.yaml









from huggingface_hub import HfApi
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi()

files_to_upload = [
    "ForenSynths.part.aa",
    "ForenSynths.part.ab",
]

# 上传文件
for filename in files_to_upload:
    print(f"Uploading {filename}...")
    response = api.upload_file(
        path_or_fileobj=f"./{filename}",
        path_in_repo=f"{filename}",
        repo_id="nebula/dfbenchmark",
        repo_type="dataset"
    )





accelerate launch --num_processes=1 --num_machines=1 --mixed_precision='bf16' -m hcpdiff.train_ac --cfg cfgs/train/ft_playground.yaml

import os
from PIL import Image
from tqdm import tqdm
folder_path = '/home/jwang/ybwork/data/dbv2/all/'
webp_files = [f for f in os.listdir(folder_path) if f.endswith('.webp')]
total_files = len(webp_files)
for filename in tqdm(webp_files, desc="Checking webp files", unit="file"):
    file_path = os.path.join(folder_path, filename)
    try:
        with Image.open(file_path) as img:
            img.verify()
    except (IOError, SyntaxError) as e:
        print('Broken file:', file_path)
        os.remove(file_path)


#  upload a folder
from huggingface_hub import HfApi
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi()

api.upload_folder(
    folder_path="./DFLIP",
    path_in_repo="./",
    repo_id="nebula/Danbooru2023-WEBP",
    repo_type="dataset",
)

