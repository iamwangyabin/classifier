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



#################################final for artists caption###############################

import os
from caption import make_caption_from_id
from db import load_db

db = load_db("danbooru2023.db")

root_dir = "/data/jwang/db2023/2kartits/"

for artist_folder in os.listdir(root_dir):
    for item in os.listdir(os.path.join(root_dir, artist_folder)):
        print(os.path.join(root_dir, artist_folder, item))
        id = int(os.path.basename(item).split('.')[0])
        # if os.path.exists(os.path.join(root_dir, artist_folder, str(id) + '.txt')):
        #     pass
        # else:
        caption = make_caption_from_id(id, keep_seperator = " ")
        with open(os.path.join(root_dir, artist_folder, str(id) + '.txt'), 'w') as file:
            file.write(caption)






#################################good picture without artists tag###############################


import math
import pickle
import os
import csv
from caption import make_caption_from_id
from db import load_db

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

db = load_db("danbooru2023.db")
source_path = '/data/jwang/db2023/target_directory'
csv_output_path = 'output.csv'

# Open the CSV file for writing
with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['image_path', 'text_zh'])
    for key, value in all_dict.items():
        if value > 1.05:
            img_path = os.path.join(source_path, f"{key}.webp")
            if os.path.exists(img_path):
                try:
                    caption = make_caption_from_id(key, keep_seperator="")
                    csvwriter.writerow([img_path, caption])
                except Exception as e:
                    print(f"Error processing image {key}: {str(e)}")
                    continue


db.close()

import os
import csv

source_path = '/data/jwang/db2023/hsr'
csv_output_path = 'hsr.csv'

with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['image_path', 'text_zh'])
    for person in os.listdir(source_path):
        for imgs in os.listdir(os.path.join(source_path, person, 'train', '1_1girl')):
            if imgs.endswith('.webp'):
                img_path = os.path.join(source_path, person, 'train', '1_1girl', imgs)
                txt_filename = os.path.splitext(imgs)[0] + '.txt'
                txt_path = os.path.join(source_path, person, 'train', '1_1girl', txt_filename)
                with open(txt_path, 'r', encoding='utf-8') as txtfile:
                    caption = txtfile.read().strip()
                csvwriter.writerow([img_path, caption])








from huggingface_hub import HfApi
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi(token="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")

files_to_upload = [
    # "sd15.part_00",
    "sd15.part_01",
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
    "clipL14openai_next_to_last_progan_train_multicls_0_real_features.parquet",
    "clipL14openai_next_to_last_progan_train_multicls_1_fake_features.parquet",

]

for i in diff_folders:
    try:
        api.upload_folder(
            folder_path=os.path.join('../preprocess/', i),
            path_in_repo=os.path.join('../preprocess/', i),
            repo_id="nebula/3mdb",
            repo_type="model",
        )
    except:
        print('failed')


