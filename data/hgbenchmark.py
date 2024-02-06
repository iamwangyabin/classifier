# AntifakePrompt DFFD ForenSynths DiffusionForensics Ojha
import pandas as pd
import os
import io
import re
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datasets import Dataset, DatasetDict

def find_images(base_path):
    image_pattern = re.compile(r'.*\.(jpg|jpeg|png|bmp)$', re.IGNORECASE)
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if image_pattern.match(file):
                image_paths.append(os.path.join(root, file))
    return image_paths


def image_to_byte_array(image_path):
    with Image.open(image_path) as image:
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr

def process_image(image_label_tuple):
    path, label = image_label_tuple
    return image_to_byte_array(path), label

def process_AntifakePrompt(data_path):
    names = []
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels = [0] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [0] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    AdvAtk_fake = find_images(os.path.join(data_path, 'AdvAtk_Imagenet'))
    labels += [1] * len(AdvAtk_fake)
    names += ['AdvAtk'] * len(AdvAtk_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [2] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [2] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    DALLE2_fake = (find_images(os.path.join(data_path, 'DALLE2', 'commonFake_COCO')) +
                   find_images(os.path.join(data_path, 'DALLE2', 'commonFake_Flickr')))
    labels += [3] * len(DALLE2_fake)
    names += ['DALLE2'] * len(DALLE2_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [4] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [4] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    Deeperforensics_fake = find_images(os.path.join(data_path, 'deeperforensics_faceOnly'))
    labels += [5] * len(Deeperforensics_fake)
    names += ['Deeperforensics'] * len(Deeperforensics_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [6] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [6] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    IF_fake = (find_images(os.path.join(data_path, 'IF', 'commonFake_COCO')) +
               find_images(os.path.join(data_path, 'IF', 'commonFake_Flickr')))
    labels += [7] * len(IF_fake)
    names += ['IF'] * len(IF_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [8] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [8] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    lteSR4_fake = find_images(os.path.join(data_path, 'lte_SR4_224'))
    labels += [9] * len(lteSR4_fake)
    names += ['lteSR4'] * len(lteSR4_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [10] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [10] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    SD2Inpaint_fake = find_images(os.path.join(data_path, 'SD2Inpaint_224'))
    labels += [11] * len(SD2Inpaint_fake)
    names += ['SD2Inpaint'] * len(SD2Inpaint_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [12] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [12] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    SDXL_fake = (find_images(os.path.join(data_path, 'SDXL', 'commonFake_COCO')) +
                 find_images(os.path.join(data_path, 'SDXL', 'commonFake_Flickr')))
    labels += [13] * len(SDXL_fake)
    names += ['SDXL'] * len(SDXL_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [14] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [14] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    Backdoor_fake = find_images(os.path.join(data_path, 'Backdoor_Imagenet'))
    labels += [15] * len(Backdoor_fake)
    names += ['Backdoor'] * len(Backdoor_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [16] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [16] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    Control_fake = find_images(os.path.join(data_path, 'Control_COCO'))
    labels += [17] * len(Control_fake)
    names += ['Control'] * len(Control_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [18] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [18] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    DataPoison_fake = find_images(os.path.join(data_path, 'DataPoison_Imagenet'))
    labels += [19] * len(DataPoison_fake)
    names += ['DataPoison'] * len(DataPoison_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [20] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [20] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    Lama_fake = find_images(os.path.join(data_path, 'lama_224'))
    labels += [21] * len(Lama_fake)
    names += ['Lama'] * len(Lama_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [22] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [22] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    SD2_fake = (find_images(os.path.join(data_path, 'SD2', 'commonFake_COCO')) +
                find_images(os.path.join(data_path, 'SD2', 'commonFake_Flickr')))
    labels += [23] * len(SD2_fake)
    names += ['SD2'] * len(SD2_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [24] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [24] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    SD2SuperRes_fake = find_images(os.path.join(data_path, 'SD2SuperRes_SR4_224'))
    labels += [25] * len(SD2SuperRes_fake)
    names += ['SD2SuperRes'] * len(SD2SuperRes_fake)
    ################################################################################
    COCO_real = find_images(os.path.join(data_path, 'COCO'))
    labels += [26] * len(COCO_real)
    names += ['COCO'] * len(COCO_real)
    Flickr_real = find_images(os.path.join(data_path, 'flickr30k_224'))
    labels += [26] * len(Flickr_real)
    names += ['Flickr'] * len(Flickr_real)
    #
    SGXL_fake = find_images(os.path.join(data_path, 'SGXL'))
    labels += [27] * len(SGXL_fake)
    names += ['SGXL'] * len(SGXL_fake)
    ################################################################################
    image_paths= (COCO_real + Flickr_real + AdvAtk_fake +
                  COCO_real + Flickr_real + DALLE2_fake +
                  COCO_real + Flickr_real + Deeperforensics_fake +
                  COCO_real + Flickr_real + IF_fake +
                  COCO_real + Flickr_real + lteSR4_fake +
                  COCO_real + Flickr_real + SD2Inpaint_fake +
                  COCO_real + Flickr_real + SDXL_fake +
                  COCO_real + Flickr_real + Backdoor_fake +
                  COCO_real + Flickr_real + Control_fake +
                  COCO_real + Flickr_real + DataPoison_fake +
                  COCO_real + Flickr_real + Lama_fake +
                  COCO_real + Flickr_real + SD2_fake +
                  COCO_real + Flickr_real + SD2SuperRes_fake +
                  COCO_real + Flickr_real + SGXL_fake)
    image_label_pairs = zip(image_paths, labels)
    cores = cpu_count()
    with Pool(cores) as pool:
        results = list(tqdm(pool.imap(process_image, image_label_pairs), total=len(image_paths)))
    image_data, labels = zip(*results)
    df = pd.DataFrame({
        'image': image_data,
        'label': labels,
        'name': names
    })
    return df

AntDF = process_AntifakePrompt('/home/jwang/ybwork/data/deepfake_benchmark/AntifakePrompt')

def ForenSynths_likedataset(data_root, sub_names, multicalss_idx):
    image_pathes = []
    labels = []
    names = []
    label_mapping = {'0_real': 0, '1_fake': 1}
    image_extensions = ('.jpg', '.jpeg', '.png')
    for id, subfolder in enumerate(sub_names):
        if multicalss_idx[id]:
            classes = os.listdir(os.path.join(data_root, subfolder))
        else:
            classes = ['']
        for cls in classes:
            root = os.path.join(data_root, subfolder, cls)
            for label in ['0_real', '1_fake']:
                label_dir = os.path.join(root, label)
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    if img_path.lower().endswith(image_extensions):
                        image_pathes.append(img_path)
                        labels.append(label_mapping[label] + id * 2)
                        names.append(subfolder)
    image_label_pairs = zip(image_pathes, labels)
    cores = cpu_count()
    with Pool(cores) as pool:
        results = list(tqdm(pool.imap(process_image, image_label_pairs), total=len(image_pathes)))
    image_data, labels = zip(*results)
    df = pd.DataFrame({
        'image': image_data,
        'label': labels,
        'name': names
    })
    return df


ForenSynthsDF = ForenSynths_likedataset('/home/jwang/ybwork/data/deepfake_benchmark/ForenSynths/test',
                        ["biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle", "progan", "san", "seeingdark", "stargan", "stylegan", "stylegan2", "whichfaceisreal"] ,
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0])

DiffusionForensicsDF = ForenSynths_likedataset('/home/jwang/ybwork/data/deepfake_benchmark/DiffusionForensics',
                        ["adm", "ddpm", "diff-stylegan", "if", "midjourney", "projectedgan", "sdv1_new2", "stylegan_official", "dalle2", "diff-projectedgan", "iddpm", "ldm", "pndm", "sdv1_new", "sdv2", "vqdiffusion"],
                        [0]*16)

OjhaDF = ForenSynths_likedataset('/home/jwang/ybwork/data/deepfake_benchmark/Ojha',
                                         ["dalle", "glide_100_10", "glide_100_27", "glide_50_27", "guided", "ldm_100", "ldm_200", "ldm_200_cfg"],
                                         [0, 0, 0, 0, 0, 0, 0, 0])

ddict = DatasetDict({
    "AntifakePrompt": Dataset.from_pandas(AntDF),
    "ForenSynths": Dataset.from_pandas(ForenSynthsDF),
    "DiffusionForensics": Dataset.from_pandas(DiffusionForensicsDF),
    "Ojha": Dataset.from_pandas(OjhaDF),
})


ddict.push_to_hub("nebula/DFBenchmarkPNG")




# Celeb-DF-v1  Celeb-DF-v2  DFDCP DFFD UADFV
#
#
# FaceForensics++ (FF++)
# DeepFakeDetection (DFD)
# DeepFake Detection Challenge (DFDC)
# FaceShifter (Fsh)
# DeeperForensics-1.0 (DF-1.0)















