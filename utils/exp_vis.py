
# 一张图
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('Ojha_224.pkl', 'rb') as file:
    loaded_dict = pickle.load(file)

session_name = 'DiffusionForensics imagenet_sdv1'

logits = loaded_dict[session_name]['y_logits']
true_labels = loaded_dict[session_name]['y_true']
logits = 1 / (1 + np.exp(-logits))
min_logit = np.min(logits)
max_logit = np.max(logits)
# max_logit=0.01
bins = 500
hist_range = (min_logit, max_logit)


plt.figure(figsize=(10, 6))
plt.hist(logits[true_labels == 0], bins=bins, range=hist_range, alpha=0.7, label='Real Label: 0', color='blue')
plt.hist(logits[true_labels == 1], bins=bins, range=hist_range, alpha=0.7, label='Fake Label: 1', color='red')

plt.title(session_name)
plt.xlabel('Logits')
plt.ylabel('Count')
plt.legend()

text = f"ap: {loaded_dict['DiffusionForensics imagenet_sdv1']['ap']:.4f}, " \
       f"r_acc0: {loaded_dict['DiffusionForensics imagenet_sdv1']['r_acc0']:.4f}, " \
       f"f_acc0: {loaded_dict['DiffusionForensics imagenet_sdv1']['f_acc0']:.4f}, " \
       f"acc0: {loaded_dict['DiffusionForensics imagenet_sdv1']['acc0']:.4f}, \n " \
       f"r_acc1: {loaded_dict['DiffusionForensics imagenet_sdv1']['r_acc1']:.4f}, " \
       f"f_acc1: {loaded_dict['DiffusionForensics imagenet_sdv1']['f_acc1']:.4f}, " \
       f"acc1: {loaded_dict['DiffusionForensics imagenet_sdv1']['acc1']:.4f}, " \
       f"best_thres: {loaded_dict['DiffusionForensics imagenet_sdv1']['best_thres']:.4f}"

plt.figtext(0.5, 0.01, text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()





# 所有图
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('NPR_224.pkl', 'rb') as file:
    loaded_dict = pickle.load(file)

# session_names = list(loaded_dict.keys())
session_names = [key for key in loaded_dict.keys() if not key.startswith('AntifakePrompt')]

num_rows = (len(session_names) + 7) // 8
fig, axs = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2.5))

for i in range(num_rows):
    for j in range(8):
        idx = i * 8 + j
        if idx >= len(session_names):
            break
        session_name = session_names[idx]
        logits = loaded_dict[session_name]['y_logits']
        true_labels = loaded_dict[session_name]['y_true']
        min_logit = np.min(logits)
        max_logit = np.max(logits)
        bins = 500
        hist_range = (min_logit, max_logit)
        axs[i, j].hist(logits[true_labels == 0], bins=bins, range=hist_range, alpha=0.7, color='blue')
        axs[i, j].hist(logits[true_labels == 1], bins=bins, range=hist_range, alpha=0.7, color='red')
        axs[i, j].set_title(session_name, fontsize=8)
        axs[i, j].tick_params(axis='both', which='major', labelsize=6)
        text = f"ap: {loaded_dict[session_name]['ap']:.2f}, " \
               f"r_acc0: {loaded_dict[session_name]['r_acc0']:.2f}, " \
               f"f_acc0: {loaded_dict[session_name]['f_acc0']:.2f}, " \
               f"acc0: {loaded_dict[session_name]['acc0']:.2f}, \n" \
               f"r_acc1: {loaded_dict[session_name]['r_acc1']:.2f}, " \
               f"f_acc1: {loaded_dict[session_name]['f_acc1']:.2f}, " \
               f"acc1: {loaded_dict[session_name]['acc1']:.2f}, " \
               f"best_thres: {loaded_dict[session_name]['best_thres']:.2f}"
        axs[i, j].text(0.5, -0.5, text, ha='center', fontsize=5, transform=axs[i, j].transAxes)

plt.tight_layout()
plt.savefig('NPR.pdf', format='pdf', bbox_inches='tight')

plt.show()


# 查看一下不同阈值的精度
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc

with open('Ojha_224.pkl', 'rb') as file:
    loaded_dict = pickle.load(file)

session_names = [key for key in loaded_dict.keys() if not key.startswith('AntifakePrompt')]

for session_name in session_names:
    y_pred = loaded_dict[session_name]['y_logits']
    y_pred = 1 / (1 + np.exp(-y_pred))
    y_true = loaded_dict[session_name]['y_true']
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.006)
    print(session_name)
    print(f"ACC: {acc0:.4f},\tR_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}")
