# 一张图
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap.umap_ as umap
from sklearn.manifold import MDS, Isomap, TSNE

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['pdf.fonttype'] = 42  # 确保PDF保存时嵌入字体

with open('clip_features_test.pkl', 'rb') as file:
    loaded_dict = pickle.load(file)

session_names = [key for key in loaded_dict.keys() if not key.startswith('AntifakePrompt')]


def scatter_subplot(ax, reduced_features, labels, title):
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        indices = labels == label
        ax.scatter(reduced_features[indices, 0], reduced_features[indices, 1], c=[color], label=label, alpha=0.5)
    ax.set_title(title)

session_name = 'Ojha ldm_200_cfg'
features = loaded_dict[session_name][0]
labels = loaded_dict[session_name][1]
tsne = TSNE(n_components=2, random_state=0)
tsne_reduced_features = tsne.fit_transform(features)
lda = LDA(n_components=1)
lda_reduced_features = lda.fit_transform(features, labels).flatten()
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
ax1 = fig.add_subplot(3, 2, 1)  #  rows,  column, 1st subplot
scatter_subplot(ax1, tsne_reduced_features, labels, 't-SNE visualization')
ax3 = fig.add_subplot(3, 2, 2)
scatter_subplot(ax3, umap_reduced_features, labels, 'UMAP visualization')
ax4 = fig.add_subplot(3, 2, 4)
scatter_subplot(ax4, pca_reduced_features, labels, 'PCA visualization')
ax5 = fig.add_subplot(3, 2, 5)
scatter_subplot(ax5, mds_reduced_features, labels, 'MDS visualization')
ax6 = fig.add_subplot(3, 2, 6)
scatter_subplot(ax6, isomap_reduced_features, labels, 'Isomap visualization')

ax2 = fig.add_subplot(3, 2, 3)
lda_jitter_y = 0.1 * np.random.rand(len(lda_reduced_features)) - 0.05
unique_labels = np.unique(labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    indices = labels == label
    ax2.scatter(lda_reduced_features[indices], lda_jitter_y[indices], c=[color], label=label, alpha=0.5)



ax2.set_title('LDA visualization')
ax2.legend()

plt.tight_layout()
plt.savefig("VisTSNE_"+session_name+'.pdf', bbox_inches='tight')



# for session_name in session_names:
#     features = loaded_dict[session_name][0]
#     labels = loaded_dict[session_name][1]
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
#
#     ax2 = fig.add_subplot(3, 2, 3)
#     lda_jitter_y = 0.1 * np.random.rand(len(lda_reduced_features)) - 0.05
#     unique_labels = np.unique(labels)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
#     for label, color in zip(unique_labels, colors):
#         indices = labels == label
#         ax2.scatter(lda_reduced_features[indices], lda_jitter_y[indices], c=[color], label=label, alpha=0.5)
#
#     ax2.set_title('LDA visualization')
#     ax2.legend()
#
#     plt.tight_layout()
#     plt.savefig("VisTSNE_"+session_name+'.pdf', bbox_inches='tight')
#
























































