import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from copy import deepcopy
from datasets import load_dataset
import io
import csv
from PIL import Image
from networks.UniversalFakeDetect.clip_models import CLIPModel, CLIPModel_inc


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"
    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

def validate(model, loader, set_name=None, sub_names=None):
    with torch.no_grad():
        y_true, y_pred = [], []
        print("Length of dataset: %d" % (len(loader)))
        for sample in loader:
            img, label = sample['image'], sample['label']
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    all_results = []
    for i, sub_task in enumerate(sub_names):
        mask = (y_true >= i * 2) & (y_true <= 1 + i * 2)
        idxes = np.where(mask)[0]
        if len(idxes) == 0:
            continue
        ap = average_precision_score(y_true[idxes] % 2, y_pred[idxes])
        r_acc0, f_acc0, acc0 = calculate_acc(y_true[idxes] % 2, y_pred[idxes], 0.5)
        best_thres = find_best_threshold(y_true[idxes] % 2, y_pred[idxes])
        r_acc1, f_acc1, acc1 = calculate_acc(y_true[idxes] % 2, y_pred[idxes], best_thres)
        all_results.append([set_name, sub_task, ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres])
        print(f"Subtask: {sub_task}\tAP: {ap:.4f},\tAcc0: {acc0:.4f},\tAcc1: {acc1:.4f},\tBestThres: {best_thres:.4f}")
    return all_results

subfolder_names = {
    'ForenSynths': ["biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle", "progan", "san", "seeingdark", "stargan",
                   "stylegan", "stylegan2", "whichfaceisreal"],
    'DiffusionForensics': ["adm", "ddpm", "diff-stylegan", "if", "midjourney", "projectedgan", "sdv1_new2",
            "stylegan_official", "dalle2", "diff-projectedgan", "iddpm", "ldm", "pndm", "sdv1_new", "sdv2", "vqdiffusion"],
    'Ojha': ["dalle", "glide_100_10", "glide_100_27", "glide_50_27", "guided", "ldm_100", "ldm_200", "ldm_200_cfg"],
    'AntifakePrompt': ['AdvAtk', 'DALLE2', 'Deeperforensics', 'IF', 'lteSR4', 'SD2Inpaint', 'SDXL', 'Backdoor',
                       'Control', 'DataPoison', 'Lama', 'SD2', 'SD2SuperRes', 'SGXL']
    }


if __name__ == '__main__':
    model = CLIPModel('ViT-L/14')
    state_dict = torch.load('networks/UniversalFakeDetect/fc_weights.pth', map_location='cpu')
    model.fc.load_state_dict(state_dict)

    # model = CLIPModel_inc('ViT-L/14')
    # state_dict = torch.load('/home/jwang/ybwork/classifier/logs/20240129_20_21_23/epoch=41-val_acc_epoch=0.84.ckpt', map_location='cpu')
    # model.fc.weight.data = state_dict['state_dict']['model.fc.weight']
    # model.fc.bias.data = state_dict['state_dict']['model.fc.bias']

    print("Model loaded..")
    model.eval()
    model.cuda()
    all_results = []
    for subdata in ['ForenSynths', 'DiffusionForensics', 'AntifakePrompt', 'Ojha']:
        dataset = load_dataset('nebula/DFBenchmarkPNG', cache_dir='/data/jwang/cache', split=subdata, num_proc=8)
        dataset.set_format("torch")
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        def custom_trans(examples):
            images = []
            keys = []
            for image, key in zip(examples["image"], examples["label"]):
                image = io.BytesIO(image)
                image = Image.open(image)
                images.append(trans(image.convert("RGB")))
                keys.append(key)
            examples['image'] = images
            examples['label'] = keys
            return examples
        dataset.set_transform(custom_trans)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=4)
        results = validate(model, data_loader, subdata, sub_names=subfolder_names[subdata])
        all_results.extend(results)

    columns = ['dataset', 'model', 'ap', 'r_acc0', 'f_acc0', 'acc0', 'r_acc1', 'f_acc1', 'acc1', 'best_thres']
    with open('model_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)
