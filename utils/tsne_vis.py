# 就是tsne分别可视化每个session的real fake分布
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# import umap.umap_ as umap
# from sklearn.manifold import MDS, Isomap, TSNE

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 12
# plt.rcParams['pdf.fonttype'] = 42  # 确保PDF保存时嵌入字体

# # with open('clip_features_test.pkl', 'rb') as file:
# #     loaded_dict = pickle.load(file)

# with open('clip_progan_train_features.pkl', 'rb') as file:
#     loaded_dict = pickle.load(file)

# session_names = [key for key in loaded_dict.keys() if not key.startswith('AntifakePrompt')]


# def scatter_subplot(ax, reduced_features, labels, title):
#     unique_labels = np.unique(labels)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
#     for label, color in zip(unique_labels, colors):
#         indices = labels == label
#         ax.scatter(reduced_features[indices, 0], reduced_features[indices, 1], c=[color], label=label, alpha=0.5)
#     ax.set_title(title)

# for session_name in session_names:
#     print(session_name)
#     features = loaded_dict[session_name][0]
#     labels = loaded_dict[session_name][1]
#     if len(labels) > 5000:
#         chosen_indices = np.random.choice(len(labels), size=5000, replace=False)
#         features = features[chosen_indices]
#         labels = labels[chosen_indices]
#     tsne = TSNE(n_components=2, random_state=0)
#     tsne_reduced_features = tsne.fit_transform(features)
#     lda = LDA(n_components=1)
#     lda_reduced_features = lda.fit_transform(features, labels).flatten()
#     umap_instance = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
#     umap_reduced_features = umap_instance.fit_transform(features)
#     pca = PCA(n_components=2, random_state=0)
#     pca_reduced_features = pca.fit_transform(features)
#     mds = MDS(n_components=2, random_state=0)
#     mds_reduced_features = mds.fit_transform(features)
#     isomap = Isomap(n_components=2)
#     isomap_reduced_features = isomap.fit_transform(features)
#     fig = plt.figure(figsize=(12, 12))
#     fig.suptitle(session_name, fontsize=24)
#     ax1 = fig.add_subplot(3, 2, 1)  #  rows,  column, 1st subplot
#     scatter_subplot(ax1, tsne_reduced_features, labels, 't-SNE visualization')
#     ax3 = fig.add_subplot(3, 2, 2)
#     scatter_subplot(ax3, umap_reduced_features, labels, 'UMAP visualization')
#     ax4 = fig.add_subplot(3, 2, 4)
#     scatter_subplot(ax4, pca_reduced_features, labels, 'PCA visualization')
#     ax5 = fig.add_subplot(3, 2, 5)
#     scatter_subplot(ax5, mds_reduced_features, labels, 'MDS visualization')
#     ax6 = fig.add_subplot(3, 2, 6)
#     scatter_subplot(ax6, isomap_reduced_features, labels, 'Isomap visualization')

#     ax2 = fig.add_subplot(3, 2, 3)
#     lda_jitter_y = 0.1 * np.random.rand(len(lda_reduced_features)) - 0.05
#     unique_labels = np.unique(labels)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
#     for label, color in zip(unique_labels, colors):
#         indices = labels == label
#         ax2.scatter(lda_reduced_features[indices], lda_jitter_y[indices], c=[color], label=label, alpha=0.5)

#     ax2.set_title('LDA visualization')
#     ax2.legend()

#     plt.tight_layout()
#     plt.savefig("VisTSNE_"+session_name+'.pdf', bbox_inches='tight')


##################################################################################################################################

# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# import umap.umap_ as umap
# from sklearn.manifold import MDS, Isomap, TSNE

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 12
# plt.rcParams['pdf.fonttype'] = 42

# with open('clip_features_test.pkl', 'rb') as file:
#     loaded_test_dict = pickle.load(file)

# with open('clip_progan_train_features.pkl', 'rb') as file:
#     loaded_train_dict = pickle.load(file)

# session_names = [key for key in loaded_test_dict.keys() if not key.startswith('AntifakePrompt')]

# def scatter_subplot(ax, reduced_features, labels, title):
#     unique_labels = np.unique(labels)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
#     for label, color in zip(unique_labels, colors):
#         indices = labels == label
#         ax.scatter(reduced_features[indices, 0], reduced_features[indices, 1], c=[color], label=label, alpha=0.5)
#     ax.set_title(title)


# train_features = loaded_train_dict['ProGAN_train'][0]
# train_labels = loaded_train_dict['ProGAN_train'][1]
# chosen_indices = np.random.choice(len(train_labels), size=1000, replace=False)
# train_features = train_features[chosen_indices]
# train_labels = train_labels[chosen_indices]

# for session_name in session_names:
#     print(session_name)
#     features = loaded_test_dict[session_name][0]
#     labels = loaded_test_dict[session_name][1]+2
#     if len(labels) > 1000:
#         chosen_indices = np.random.choice(len(labels), size=1000, replace=False)
#         features = features[chosen_indices]
#         labels = labels[chosen_indices]

#     features = np.concatenate((features, train_features), axis=0)
#     labels = np.concatenate((labels, train_labels), axis=0)
#     tsne = TSNE(n_components=2, random_state=0)
#     tsne_reduced_features = tsne.fit_transform(features)
#     lda = LDA(n_components=1)
#     lda_reduced_features = lda.fit_transform(features, labels).flatten()
#     umap_instance = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
#     umap_reduced_features = umap_instance.fit_transform(features)
#     pca = PCA(n_components=2, random_state=0)
#     pca_reduced_features = pca.fit_transform(features)
#     mds = MDS(n_components=2, random_state=0)
#     mds_reduced_features = mds.fit_transform(features)
#     isomap = Isomap(n_components=2)
#     isomap_reduced_features = isomap.fit_transform(features)
#     fig = plt.figure(figsize=(12, 12))
#     fig.suptitle(session_name, fontsize=24)
#     ax1 = fig.add_subplot(3, 2, 1)  #  rows,  column, 1st subplot
#     scatter_subplot(ax1, tsne_reduced_features, labels, 't-SNE visualization')
#     ax3 = fig.add_subplot(3, 2, 2)
#     scatter_subplot(ax3, umap_reduced_features, labels, 'UMAP visualization')
#     ax4 = fig.add_subplot(3, 2, 4)
#     scatter_subplot(ax4, pca_reduced_features, labels, 'PCA visualization')
#     ax5 = fig.add_subplot(3, 2, 5)
#     scatter_subplot(ax5, mds_reduced_features, labels, 'MDS visualization')
#     ax6 = fig.add_subplot(3, 2, 6)
#     scatter_subplot(ax6, isomap_reduced_features, labels, 'Isomap visualization')

#     ax2 = fig.add_subplot(3, 2, 3)
#     lda_jitter_y = 0.1 * np.random.rand(len(lda_reduced_features)) - 0.05
#     unique_labels = np.unique(labels)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
#     for label, color in zip(unique_labels, colors):
#         indices = labels == label
#         ax2.scatter(lda_reduced_features[indices], lda_jitter_y[indices], c=[color], label=label, alpha=0.5)

#     ax2.set_title('LDA visualization')
#     ax2.legend()

#     plt.tight_layout()
#     plt.savefig("VisCompare_"+session_name+'.pdf', bbox_inches='tight')


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap.umap_ as umap
from sklearn.manifold import MDS, Isomap, TSNE

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['pdf.fonttype'] = 42

with open('dino_test_features.pkl', 'rb') as file:
    loaded_test_dict = pickle.load(file)

with open('dino_progan_train_features.pkl', 'rb') as file:
    loaded_train_dict = pickle.load(file)

session_names = [key for key in loaded_test_dict.keys() if not key.startswith('AntifakePrompt')]


def scatter_subplot(ax, reduced_features, labels, title):
    # unique_labels = np.unique(labels)
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    unique_labels = np.unique(labels)
    num_cold = 2
    num_warm = len(unique_labels) - num_cold
    cold_colors = plt.cm.Reds(np.linspace(0.9, 0.5, num_cold))
    warm_colors = plt.cm.Greys(np.linspace(0.35, 0.9, num_warm))
    colors = np.zeros((len(unique_labels), 4))  # 4是因为颜色是RGBA格式
    for i, label in enumerate(unique_labels):
        if label == 0 or label == 1:
            colors[i] = cold_colors[label]
        else:
            warm_index = label - num_cold
            colors[i] = warm_colors[warm_index]

    for label, color in zip(unique_labels, colors):
        indices = labels == label
        ax.scatter(reduced_features[indices, 0], reduced_features[indices, 1], c=[color], label=label, alpha=0.5)
    ax.set_title(title)


train_features = loaded_train_dict['ProGAN_train'][0]
train_labels = loaded_train_dict['ProGAN_train'][1]

chosen_indices = np.random.choice(len(train_labels), size=5000, replace=False)
train_features = train_features[chosen_indices]
train_labels = train_labels[chosen_indices]

for id, session_name in enumerate(session_names):
    print(session_name)
    if session_name.split(' ')[0] == 'DiffusionForensics':
        id = 1
    elif session_name.split(' ')[0] == 'AIGCDetect':
        id = 2
    elif session_name.split(' ')[0] == 'ForenSynths':
        id = 3
    elif session_name.split(' ')[0] == 'Ojha':
        id = 4
    features = loaded_test_dict[session_name][0]
    labels = loaded_test_dict[session_name][1] + id * 2
    if len(labels) > 100:
        chosen_indices = np.random.choice(len(labels), size=100, replace=False)
        features = features[chosen_indices]
        labels = labels[chosen_indices]
    train_features = np.concatenate((features, train_features), axis=0)
    train_labels = np.concatenate((labels, train_labels), axis=0)

# even_indices = [i for i, label in enumerate(train_labels) if label % 2 == 0]
# even_train_features = [train_features[i] for i in even_indices]
# even_train_labels = [int(label/2) for label in train_labels if label % 2 == 0]
# # even_train_labels = [0 for label in train_labels if label % 2 == 0]

# odd_indices = [i for i, label in enumerate(train_labels) if label % 2 != 0]
# odd_train_features = [train_features[i] for i in odd_indices]
# odd_train_labels = [int(label/2)+56 for label in train_labels if label % 2 != 0]
# # odd_train_labels = [1 for label in train_labels if label % 2 != 0]

# features = np.array(even_train_features + odd_train_features)
# labels = np.array(even_train_labels + odd_train_labels)

features = train_features
labels = train_labels

tsne = TSNE(n_components=2, random_state=0)
tsne_reduced_features = tsne.fit_transform(features)
# lda = LDA(n_components=1)
# lda_reduced_features = lda.fit_transform(features, labels).flatten()
umap_instance = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
umap_reduced_features = umap_instance.fit_transform(features)
pca = PCA(n_components=2, random_state=0)
pca_reduced_features = pca.fit_transform(features)
mds = MDS(n_components=2, random_state=0)
mds_reduced_features = mds.fit_transform(features)
isomap = Isomap(n_components=2)
isomap_reduced_features = isomap.fit_transform(features)
fig = plt.figure(figsize=(12, 12))
fig.suptitle(session_name, fontsize=24)
ax1 = fig.add_subplot(3, 2, 1)  # rows,  column, 1st subplot
scatter_subplot(ax1, tsne_reduced_features, labels, 't-SNE visualization')
ax3 = fig.add_subplot(3, 2, 2)
scatter_subplot(ax3, umap_reduced_features, labels, 'UMAP visualization')
ax4 = fig.add_subplot(3, 2, 3)
scatter_subplot(ax4, pca_reduced_features, labels, 'PCA visualization')
ax5 = fig.add_subplot(3, 2, 4)
scatter_subplot(ax5, mds_reduced_features, labels, 'MDS visualization')
ax6 = fig.add_subplot(3, 2, 5)
scatter_subplot(ax6, isomap_reduced_features, labels, 'Isomap visualization')

# ax2 = fig.add_subplot(3, 2, 3)
# lda_jitter_y = 0.1 * np.random.rand(len(lda_reduced_features)) - 0.05
# unique_labels = np.unique(labels)
# colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
# for label, color in zip(unique_labels, colors):
#     indices = labels == label
#     ax2.scatter(lda_reduced_features[indices], lda_jitter_y[indices], c=[color], label=label, alpha=0.5)
# ax2.set_title('LDA visualization')
ax4.legend()

plt.tight_layout()
plt.savefig('all_compare.pdf', bbox_inches='tight')










# 只可视化progan训练集的real图片特征，看看能不能发现类别不同：



import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.manifold import MDS, Isomap, TSNE

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['pdf.fonttype'] = 42


with open('clip_progan_train_multicls_fake_features.pkl', 'rb') as file:
    loaded_train_dict = pickle.load(file)

def scatter_subplot(ax, reduced_features, labels, title):
    unique_labels = np.unique(labels)
    num_cold = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0.9, 0.1, num_cold))
    for label, color in zip(unique_labels, colors):
        indices = labels == label
        ax.scatter(reduced_features[indices, 0], reduced_features[indices, 1], c=[color], label=label, alpha=0.7)
    ax.set_title(title)


train_features = loaded_train_dict['ProGAN_train_fake'][0]
train_labels = loaded_train_dict['ProGAN_train_fake'][1]

chosen_indices = np.random.choice(len(train_labels), size=5000, replace=False)
train_features = train_features[chosen_indices]
train_labels = train_labels[chosen_indices]

features = train_features
labels = train_labels

tsne = TSNE(n_components=2, random_state=0)
tsne_reduced_features = tsne.fit_transform(features)
umap_instance = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
umap_reduced_features = umap_instance.fit_transform(features)
mds = MDS(n_components=2, random_state=0)
mds_reduced_features = mds.fit_transform(features)

fig = plt.figure(figsize=(12, 12))
fig.suptitle(session_name, fontsize=24)
ax1 = fig.add_subplot(2, 2, 1)
scatter_subplot(ax1, tsne_reduced_features, labels, 't-SNE visualization')
ax3 = fig.add_subplot(2, 2, 2)
scatter_subplot(ax3, umap_reduced_features, labels, 'UMAP visualization')
ax4 = fig.add_subplot(2, 2, 3)
scatter_subplot(ax4, mds_reduced_features, labels, 'PCA visualization')

ax4.legend()

plt.tight_layout()
plt.savefig('progan_fake.pdf', bbox_inches='tight')












