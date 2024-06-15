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


data = {}
with open("/data/jwang/db2023/posts.json", "r") as file:
    for line in file:
        try:
            json_obj = json.loads(line)
            data[json_obj['id']] = json_obj
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
    tags_general = process_tags(data_item.get("tag_string_general", ""))
    tags_character = process_tags(data_item.get("tag_string_character", ""))
    if tags_character == "original":
        tags_character = ""
    else:
        tags_character = tags_character
    tags_artist = data_item.get("tag_string_artist", "")
    tags_meta = process_tags(data_item.get("tag_string_meta", ""))
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
    pre_separator_str = ', '.join(pre_separator_tags)
    post_separator_str = ', '.join(post_separator_tags)
    caption = f"{pre_separator_str}, {post_separator_str}"
    return caption


img_dir = "/data/jwang/db2023/target_directory"

for img in tqdm(os.listdir(img_dir)):
    try:
        caption = generate_tags(data[int(img.split('.')[0])])
    except:
        caption = ""
    with open(os.path.join(img_dir, img.split('.')[0]+'.txt'), 'w') as file:
        file.write(caption)


#############################################选图######################################################

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

# data = {}
artistname2ID={}
with open("./posts.json", "r", encoding='utf-8') as file:
    for line in file:
        try:
            json_obj = json.loads(line)
            # data[json_obj['id']] = json_obj
            if json_obj['tag_string_artist'] in artistname2ID:
                artistname2ID[json_obj['tag_string_artist']].append(json_obj['id'])
            else:
                artistname2ID[json_obj['tag_string_artist']] = [json_obj['id']]
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")


with open('danbooru_tag_from_pixiv_top_artists_total.json', 'r', encoding='utf-8') as file:
    artscore = json.load(file)

# 选中的艺术家
items_with_scores = [(key, value['pixiv_hot_score']) for key, value in artscore.items()]
sorted_items = sorted(items_with_scores, key=lambda x: x[1], reverse=True)
top_10000_items = sorted_items[:10000]
top_10000_keys = [item[0] for item in top_10000_items]
top_10000_full_items = [artscore[key]['danbooru_info'][0]['name'] for key in top_10000_keys]



# 查看下图片的美学指标（简单筛选下）

directory = '/home/jwang/ybwork/nai3/'
all_results = []

pickle_files = [f for f in os.listdir(directory) if f.endswith('.pickle')]

pickle_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

for filename in pickle_files:
    filepath = os.path.join(directory, filename)
    with open(filepath, 'rb') as file:
        results = pickle.load(file)
        if isinstance(results, list):
            all_results.extend(results)
        else:
            print(f"Error: {filename} does not contain a list.")

all_dict = {}
weights = [3, 3, 1, 1, 0, 0, 0]


for item in all_results:
    sample_id, logits_str = item
    logits = logits_str.split(',')
    logits = [float(logit) for logit in logits]
    sigmoid_logits = [1 / (1 + math.exp(-logit)) for logit in logits]
    exp_sigmoid_logits = [math.exp(sigmoid_logit) for sigmoid_logit in sigmoid_logits]
    sum_exp_sigmoid_logits = sum(exp_sigmoid_logits)
    softmax_sigmoid_logits = [exp_sigmoid_logit / sum_exp_sigmoid_logits for exp_sigmoid_logit in exp_sigmoid_logits]
    final_score = sum(weight * softmax_sigmoid_logit for weight, softmax_sigmoid_logit in zip(weights, softmax_sigmoid_logits))
    all_dict[int(sample_id.split('.')[0])] = final_score

top_10000_artist_images = {}
totol_num = 0
for item in top_10000_full_items:
    if item in artistname2ID:
        temp = []
        for id in artistname2ID[item]:
            if id in all_dict and all_dict[id] >0.95:
                temp.append(id)
        if len(temp) >= 200:
            top_10000_artist_images[item] = temp
            totol_num += len(temp)

top_10000_artist_images['putong_xiao_gou'] = artistname2ID['putong_xiao_gou']



import shutil

source_path = '/data/jwang/db2023/target_directory'
destination_path = '/data/jwang/db2023/2kartits'

for artist, images in top_10000_artist_images.items():
    os.makedirs(os.path.join(destination_path, artist), exist_ok=True)
    for item in images:
        shutil.copy(os.path.join(source_path, str(item)+'.webp'), os.path.join(destination_path, artist, str(item)+'.webp'))









from huggingface_hub import HfApi
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi(token="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")

files_to_upload = [
    "clipbased.tar",
]

for filename in files_to_upload:
    print(f"Uploading {filename}...")
    response = api.upload_file(
        path_or_fileobj=f"./{filename}",
        path_in_repo=f"{filename}",
        repo_id="nebula/dfbenchmark",
        repo_type="dataset"
    )


#  upload a folder
from huggingface_hub import HfApi
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi()

diff_folders = [
    "arpaug-ViTL-a01b1c1-881_20240514_08_32_18",
    "arpaug-ViTL-a05b05c1-881_20240514_02_58_41",
    "arpaug-ViTL-a05b1c1-881_20240514_07_23_24",
    "arpaug-ViTL-a0b1c1-881_20240514_02_47_45",
    "arpaug-ViTL-a1b05c1-881_20240514_06_15_04",
    "arpaug-ViTL-a1b0c1-881_20240514_02_30_25",
    "arpaug-ViTL-a1b1c1-881-first12cls_20240514_03_28_50",
    "arpaug-ViTL-a1b1c1-881-first16cls_20240514_04_11_21",
    "arpaug-ViTL-a1b1c1-881-first2cls_20240511_23_01_23",
    "arpaug-ViTL-a1b1c1-881-first4cls_20240511_22_43_55",
    "arpaug-ViTL-a1b1c1-881-first8cls_20240511_23_12_42",
    "arpaug-ViTL-a2b2c1-881_20240514_05_06_45"
]

for i in diff_folders:
    try:
        api.upload_folder(
            folder_path=os.path.join('../preprocess/', i),
            path_in_repo=os.path.join('../preprocess/', i),
            repo_id="nebula/testmodel",
            repo_type="model",
        )
    except:
        print('failed')


#################################final for artists caption###############################

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

from caption import make_caption_from_id

from db import load_db

db = load_db("danbooru2023.db")


root_dir = "/data/jwang/db2023/2kartits/"


for artist_folder in os.listdir(root_dir):
    for item in os.listdir(os.path.join(root_dir, artist_folder)):
        print(os.path.join(root_dir, artist_folder, item))
        id = int(os.path.basename(item).split('.')[0])
        if os.path.exists(os.path.join(root_dir, artist_folder, str(id) + '.txt')):
            pass
        else:
            caption = make_caption_from_id(id)
            with open(os.path.join(root_dir, artist_folder, str(id) + '.txt'), 'w') as file:
                file.write(caption)




\begin{table}[]
\caption{Main Results.}
\centering
\small
\begin{tabular}{lrrrrrrrrrrrr}
\hline
\multicolumn{1}{c|}{\multirow{2}{*}{Method}} & \multicolumn{4}{c|}{ForenSynths~\cite{wang2020cnn}} & \multicolumn{4}{c|}{GenImage~\cite{zhu2023genimage}} & \multicolumn{4}{c}{GANGen-Detection~\cite{chuangchuangtan-GANGen-Detection}} \\ \cline{2-13}
\multicolumn{1}{c|}{} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c|}{ACC} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c|}{ACC} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c}{ACC} \\ \hline
\multicolumn{1}{l|}{CNNDet\cite{wang2020cnn}} & 89.00 & 89.53 & 54.04 & \multicolumn{1}{r|}{70.18} & 64.95 & 67.77 & 9.60 & \multicolumn{1}{r|}{52.04} & 82.69 & 82.50 & 37.28 & 62.15 \\
\multicolumn{1}{l|}{FreDect\cite{frank2020leveraging}} & 67.07 & 65.86 & 59.86 & \multicolumn{1}{r|}{61.51} & 71.34 & 77.50 & \textbf{53.77} & \multicolumn{1}{r|}{{\ul 63.71}} & 62.54 & 63.12 & 21.10 & 53.51 \\
\multicolumn{1}{l|}{GramNet\cite{liu2020global}} & 59.37 & 60.66 & 13.27 & \multicolumn{1}{r|}{50.05} & 66.70 & 69.92 & 26.73 & \multicolumn{1}{r|}{58.18} & 52.12 & 52.60 & 0.05 & 50.01 \\
\multicolumn{1}{l|}{Fusing\cite{ju2022fusing}} & 91.21 & 91.18 & 42.63 & \multicolumn{1}{r|}{65.84} & 73.88 & 76.04 & 6.03 & \multicolumn{1}{r|}{51.64} & 90.55 & 90.46 & 58.54 & 71.55 \\
\multicolumn{1}{l|}{LNP\cite{liu2022detecting}} & 66.31 & 68.20 & 12.50 & \multicolumn{1}{r|}{52.19} & 57.09 & 56.04 & 8.79 & \multicolumn{1}{r|}{51.84} & 63.03 & 64.51 & 3.49 & 50.70 \\
\multicolumn{1}{l|}{SPrompts\cite{wang2022s}} & 90.06 & 89.35 & 32.33 & \multicolumn{1}{r|}{60.60} & 55.58 & 53.37 & 21.80 & \multicolumn{1}{r|}{53.69} & 48.23 & 46.74 & 47.65 & 49.97 \\
\multicolumn{1}{l|}{UnivFD\cite{ojha2023towards}} & {\ul 92.96} & {\ul 92.31} & {\ul 60.49} & \multicolumn{1}{r|}{{\ul 74.69}} & {\ul 77.59} & {\ul 78.65} & 30.39 & \multicolumn{1}{r|}{60.22} & {\ul 91.91} & {\ul 91.26} & {\ul 79.95} & {\ul 83.68} \\
\multicolumn{1}{l|}{LGrad\cite{tan2023learning}} & 55.01 & 55.31 & 4.62 & \multicolumn{1}{r|}{48.68} & 54.64 & 55.29 & 1.96 & \multicolumn{1}{r|}{50.29} & 55.63 & 56.84 & 0.05 & 50.00 \\
\multicolumn{1}{l|}{NPR\cite{tan2023rethinking}  } & 48.26 & 45.22 & 12.98 & \multicolumn{1}{r|}{50.95} & 65.49 & 65.86 & 7.37 & \multicolumn{1}{r|}{51.64} & {\ul 88.83} & {\ul 87.78} & 17.15 & 54.89 \\
\multicolumn{1}{l|}{Freqnet\cite{tan2024frequencyaware}} & 51.77 & 52.24 & 5.07 & \multicolumn{1}{r|}{49.52} & 56.52 & 56.86 & 2.47 & \multicolumn{1}{r|}{50.36} & 52.10 & 53.00 & 44.75 & 52.25 \\ \hline
\multicolumn{1}{l|}{our881} & \textbf{94.09} & \textbf{94.06} & \textbf{79.21} & \textbf{80.37} & \textbf{80.98} & \textbf{82.12} & {\ul 47.77} & \textbf{68.02} & \textbf{93.22} & \textbf{93.39} & \textbf{85.79} & \textbf{84.49}  \\

\hline

 & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} \\ \hline
\multicolumn{1}{c|}{\multirow{2}{*}{Method}} & \multicolumn{4}{c|}{DiffusionForensics~\cite{wang2023dire}} & \multicolumn{4}{c|}{Ojha~\cite{ojha2023towards}} & \multicolumn{4}{c}{DIF~\cite{Sinitsa_2024_WACV}} \\ \cline{2-13}
\multicolumn{1}{c|}{} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c|}{ACC} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c|}{ACC} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c}{ACC} \\ \hline
\multicolumn{1}{l|}{CNNDet\cite{wang2020cnn}} & 64.70 & 69.36 & 11.15 & \multicolumn{1}{r|}{57.31} & 63.41 & 65.80 & 6.78 & \multicolumn{1}{r|}{51.31} & 74.31 & 75.16 & 34.65 & 62.50 \\
\multicolumn{1}{l|}{FreDect\cite{frank2020leveraging}} & 43.60 & 43.87 & {\ul 40.05} & \multicolumn{1}{r|}{43.20} & 66.31 & 71.39 & 57.26 & \multicolumn{1}{r|}{62.24} & 69.51 & 70.92 & {\ul 59.67} & 64.95 \\
\multicolumn{1}{l|}{GramNet\cite{liu2020global}} & 78.42 & 80.96 & 26.16 & \multicolumn{1}{r|}{60.51} & 54.76 & 54.84 & 0.23 & \multicolumn{1}{r|}{49.91} & 50.06 & 45.21 & 5.85 & 44.82 \\
\multicolumn{1}{l|}{Fusing\cite{ju2022fusing}} & 62.89 & 70.62 & 1.57 & \multicolumn{1}{r|}{56.23} & 73.61 & 73.92 & 7.71 & \multicolumn{1}{r|}{51.88} & 81.64 & 81.19 & 33.93 & 63.36 \\
\multicolumn{1}{l|}{LNP\cite{liu2022detecting}} & {\ul 84.49} & \textbf{87.63} & 22.60 & \multicolumn{1}{r|}{{\ul 60.94}} & 49.37 & 47.58 & 4.41 & \multicolumn{1}{r|}{50.44} & 55.75 & 53.46 & 6.01 & 49.16 \\
\multicolumn{1}{l|}{SPrompts\cite{wang2022s}} & 57.49 & 56.62 & 10.12 & \multicolumn{1}{r|}{53.26} & 45.08 & 36.47 & 4.95 & \multicolumn{1}{r|}{50.43} & 46.01 & 41.55 & 8.68 & 47.68 \\
\multicolumn{1}{l|}{UnivFD\cite{ojha2023towards}} & 62.42 & 65.73 & 18.39 & \multicolumn{1}{r|}{60.68} & \textbf{93.88} & \textbf{93.40} & {\ul 67.45} & \multicolumn{1}{r|}{{\ul 76.19}} & {\ul 88.31} & {\ul 88.35} & 56.39 & {\ul 73.71} \\
\multicolumn{1}{l|}{LGrad\cite{tan2023learning}} & 73.25 & 78.82 & 4.37 & \multicolumn{1}{r|}{56.69} & 44.39 & 39.01 & 0.32 & \multicolumn{1}{r|}{49.99} & 51.33 & 51.09 & 0.79 & 49.99 \\
\multicolumn{1}{l|}{NPR\cite{tan2023rethinking}} & \textbf{86.06} & {\ul 87.38} & 12.25 & \multicolumn{1}{r|}{57.20} & 79.06 & 78.38 & 4.18 & \multicolumn{1}{r|}{51.02} & 76.68 & 75.93 & 19.19 & 55.82 \\
\multicolumn{1}{l|}{Freqnet\cite{tan2024frequencyaware}} & 43.02 & 43.97 & 5.02 & \multicolumn{1}{r|}{55.18} & 54.68 & 49.09 & 1.80 & \multicolumn{1}{r|}{50.36} & 48.96 & 46.11 & 0.70 & 49.27 \\ \hline
\multicolumn{1}{l|}{our881} & 63.48 & 72.38 & \textbf{42.91} & \textbf{67.24} & {\ul 91.98} & {\ul 91.78} & \textbf{81.79} & \textbf{82.90} & \textbf{88.76} & \textbf{89.47} & \textbf{70.32} & \textbf{79.23}  \\

\hline

 & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} & \multicolumn{1}{l}{} \\ \hline
\multicolumn{1}{c|}{\multirow{2}{*}{Method}} & \multicolumn{4}{c|}{Celeb-DF-v1~\cite{Celeb_DF_cvpr20}} & \multicolumn{4}{c|}{Celeb-DF-v2~\cite{Celeb_DF_cvpr20}} & \multicolumn{4}{c}{UADFV~\cite{li2018ictu}} \\ \cline{2-13}
\multicolumn{1}{c|}{} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c|}{ACC} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c|}{ACC} & \multicolumn{1}{c}{AP} & \multicolumn{1}{c}{AUC} & \multicolumn{1}{c}{F1} & \multicolumn{1}{c}{ACC} \\ \hline
\multicolumn{1}{l|}{CNNDet\cite{wang2020cnn}} & 64.40 & 50.08 & 0.14 & \multicolumn{1}{r|}{33.94} & 87.60 & 54.81 & 0.19 & \multicolumn{1}{r|}{13.57} & 61.65 & 63.42 & 0.13 & 50.42 \\
\multicolumn{1}{l|}{FreDect\cite{frank2020leveraging}} & 72.54 & 60.38 & {\ul 53.93} & \multicolumn{1}{r|}{{\ul 51.88}} & 88.10 & 55.77 & {\ul 55.18} & \multicolumn{1}{r|}{{\ul 43.61}} & 62.42 & 65.15 & {\ul 52.11} & 60.03 \\
\multicolumn{1}{l|}{GramNet\cite{liu2020global}} & 67.29 & 52.55 & 0.04 & \multicolumn{1}{r|}{33.94} & 88.36 & 56.11 & 0.08 & \multicolumn{1}{r|}{13.53} & 39.94 & 35.92 & 0.00 & 49.77 \\
\multicolumn{1}{l|}{Fusing\cite{ju2022fusing}} & 61.87 & 46.05 & 0.21 & \multicolumn{1}{r|}{33.86} & 86.66 & 51.97 & 0.56 & \multicolumn{1}{r|}{13.70} & 53.18 & 54.43 & 0.65 & 50.49 \\
\multicolumn{1}{l|}{LNP\cite{liu2022detecting}} & 64.16 & 48.24 & 1.66 & \multicolumn{1}{r|}{34.11} & 86.76 & 50.88 & 3.30 & \multicolumn{1}{r|}{14.75} & 44.87 & 35.74 & 10.18 & 51.73 \\
\multicolumn{1}{l|}{SPrompts\cite{wang2022s}} & 66.38 & 49.87 & 1.75 & \multicolumn{1}{r|}{34.33} & 87.30 & 50.95 & 1.94 & \multicolumn{1}{r|}{14.29} & 46.20 & 48.37 & 11.72 & 48.01 \\
\multicolumn{1}{l|}{UnivFD\cite{ojha2023towards}} & \textbf{83.73} & \textbf{74.19} & 16.64 & \multicolumn{1}{r|}{39.66} & {\ul 92.49} & {\ul67.08} & 10.28 & \multicolumn{1}{r|}{18.10} & \textbf{91.50} & \textbf{91.27} & 43.08 & {\ul63.87} \\
\multicolumn{1}{l|}{LGrad\cite{tan2023learning}} & 65.98 & 51.64 & 0.14 & \multicolumn{1}{r|}{33.95} & 87.33 & 52.92 & 0.56 & \multicolumn{1}{r|}{13.71} & 44.47 & 43.54 & 0.13 & 49.90 \\
\multicolumn{1}{l|}{NPR\cite{tan2023rethinking}} & 62.57 & 45.85 & 7.28 & \multicolumn{1}{r|}{34.45} & 86.07 & 50.42 & 7.48 & \multicolumn{1}{r|}{16.23} & 46.72 & 36.31 & 10.78 & 50.42 \\
\multicolumn{1}{l|}{Freqnet\cite{tan2024frequencyaware}} & 64.66 & 50.47 & 18.85 & \multicolumn{1}{r|}{36.67} & 86.18 & 51.14 & 17.32 & \multicolumn{1}{r|}{20.41} & 44.58 & 45.22 & 11.44 & 45.05 \\
\hline
\multicolumn{1}{l|}{our441} & {\ul 83.69} & {\ul 73.77} & \textbf{71.28} & \textbf{65.93} & \textbf{93.16} & \textbf{70.19} & \textbf{76.95} & \textbf{65.69} & {\ul 85.92} & {\ul 85.35} & \textbf{79.17} & \textbf{77.86} \\

\hline
\end{tabular}
\end{table}




CNNDet\cite{wang2020cnn}
FreDect\cite{frank2020leveraging}
GramNet\cite{liu2020global}
Fusing\cite{ju2022fusing}
LNP\cite{liu2022detecting}
SPrompts\cite{wang2022s}
UnivFD\cite{ojha2023towards}
LGrad\cite{tan2023learning}
NPR\cite{tan2023rethinking}
Freqnet\cite{tan2024frequencyaware}


\begin{table}[]
\small
\begin{tabular}{c|l|rrrrrrrr}
\hline
& \multicolumn{1}{c|}{\textbf{Method}} & \multicolumn{1}{c}{$\mathbf{AP}$} & \multicolumn{1}{c}{$\mathbf{F1}$} & \multicolumn{1}{c}{$\mathbf{ACC_r}$} & \multicolumn{1}{c}{$\mathbf{ACC_f}$} & \multicolumn{1}{c}{$\mathbf{ACC}$} & \multicolumn{1}{c}{$\mathbf{AUC_{roc}}$} & \multicolumn{1}{c}{$\mathbf{AUC_{f1}}$} & \multicolumn{1}{c}{$\mathbf{AUC_{f2}}$} \\ \hline





\end{tabular}
\end{table}







