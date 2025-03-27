import argparse
import scanpy as sc
import networkx as nx
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import os
import itertools
from copy import copy

sc.settings.verbosity = 4

# -----------------------------------------------
# Dictionaries
# -----------------------------------------------

matrix_cmap = {
    'SMA_fibro_HS': '#1f77b4',
    'Hypoxia matrix': '#aec7e8',
    'Low_level_hypo': '#ffbb78',
    'CS_Bi_Col_HS_Fibro': '#2ca02c',
    'CS': '#98df8a',
    'Hypoxia_blood': '#ff9896',
    'PanCyto': '#9467bd',
    'Brevican_Neurocan': '#c5b0d5',
    'Vimentin_Aggre': '#8c564b',
    'Brevican': '#c49c94',
    'GLUT1': '#e377c2',
    'Inflammatory hypoxia': '#f7b6d2',
    'Vers_Neu_Brev': '#9edae5',
    'TNC_Brevican': '#bcbd22',
    'Artifact': '#000000',
    'HA': '#17becf'
}

pixie_envs = {
    1: 'SMA_fibro_HS',
    2: 'Hypoxia matrix',
    4: 'Low_level_hypo',
    5: 'CS_Bi_Col_HS_Fibro',
    6: 'CS',
    8: 'Hypoxia_blood',
    9: 'PanCyto',
    10: 'Brevican_Neurocan',
    11: 'Vimentin_Aggre',
    12: 'Brevican',
    13: 'GLUT1',
    14: 'Inflammatory hypoxia',
    15: 'Vers_Neu_Brev',
    17: 'TNC_Brevican',
    29: 'Artifact',
    30: 'HA'
}

pixie_cmap = {
    0: '#000000',
    1: '#1f77b4',
    2: '#aec7e8',
    4: '#ffbb78',
    5: '#2ca02c',
    6: '#98df8a',
    8: '#ff9896',
    9: '#9467bd',
    10: '#c5b0d5',
    11: '#8c564b',
    12: '#c49c94',
    13: '#e377c2',
    14: '#f7b6d2',
    15: '#9edae5',
    17: '#bcbd22',
    29: '#000000',
    30: '#17becf'
}

# -----------------------------------------------
# Parse arguments
# -----------------------------------------------
parser = argparse.ArgumentParser(description="Process a single npy embedding file.")
parser.add_argument("--npy", required=True, help="Path to the .npy file (e.g. node_embeddings_node2vec.npy).")
args = parser.parse_args()

# Determine name based on filename
if "node2vec" in args.npy.lower():
    name = "node2vec"
elif "graphsage" in args.npy.lower():
    name = "graphsage"
else:
    # Fallback if the filename doesn't match known patterns
    name = "embedding"

# -----------------------------------------------
# Read in networkx graphs
# -----------------------------------------------
G_list = [nx.read_gexf(f'Matrix_networkx_graphs/{f}') for f in os.listdir('Matrix_networkx_graphs')]
roi_names = [x.replace('_pixel_mask.gexf', '') for x in os.listdir('Matrix_networkx_graphs')]
roi_names = pd.Series(roi_names).astype('category')

G_nx = nx.disjoint_union_all(G_list)
for node, data in G_nx.nodes(data=True):
    if "label" in data:
        data["ecm_env"] = data.pop("label")

node_data = pd.DataFrame.from_dict(dict(G_nx.nodes(data=True)), orient="index")

list_of_lists = [[roi for x in range(g.number_of_nodes())] for roi, g in zip(roi_names, G_list)]
rois = list(itertools.chain.from_iterable(list_of_lists))
node_data['ROI'] = rois
node_data.loc[:,'pixie'] = node_data['ecm_env'].astype(int).map(pixie_envs)

roi_to_case_dict = pd.read_csv('roi_to_case.csv', index_col=0).to_dict()

# -----------------------------------------------
# Retrieve saved embeddings as Numpy array file
# -----------------------------------------------
adatas = {}

# Load embeddings
node_embeddings = np.load(args.npy)

# Create AnnData
adatas[name] = ad.AnnData(X=node_embeddings, obs=node_data)

# Map in case info
adatas[name].obs['Case'] = adatas[name].obs.ROI.map(roi_to_case_dict)
adatas[name].obs['Case'] = adatas[name].obs['Case'].astype('category')
adatas[name].obs['Case_num'] = adatas[name].obs['Case'].cat.codes

# Calculate proportion area for each node in each ROI
for r in adatas[name].obs.ROI.unique().tolist():
    node_data_roi = adatas[name].obs.loc[adatas[name].obs.ROI == r, :]
    total_area = node_data_roi.area.sum()
    adatas[name].obs.loc[adatas[name].obs.ROI == r, 'area_norm'] = node_data_roi['area'].div(total_area)

adatas[name].obs['pixie'] = adatas[name].obs['pixie'].astype('category')
adatas[name].uns['pixie_colors'] = adatas[name].obs['pixie'].cat.categories.map(matrix_cmap)

# Scanpy processing
sc.tl.pca(adatas[name], n_comps=10)
sc.external.pp.bbknn(adatas[name], batch_key='Case')
sc.tl.umap(adatas[name], min_dist=0.4)

# Save colourmaps
adatas[name].obs['pixie'] = adatas[name].obs['pixie'].astype('category')
adatas[name].obs['pixie'].cat.categories.map(matrix_cmap)

# Clustering
sc.tl.leiden(adatas[name], resolution=0.1)

# Save UMAP
sc.pl.umap(adatas[name],
           color=['pixie', 'leiden', 'Case_num', 'area_norm'],
           s=10, ncols=1, cmap='jet',
           save=f'_{name}.png')

# Plot and save heatmap
fig, ax2 = plt.subplots(2, 1, figsize=(5, 5))
plt.subplots_adjust(hspace=3)
sb.heatmap(pd.crosstab(adatas[name].obs.leiden, adatas[name].obs.pixie, normalize='index'),
           ax=ax2.flatten()[0], vmax=0.3)
ax2.flatten()[0].set_title('Norm across leidens')
sb.heatmap(pd.crosstab(adatas[name].obs.leiden, adatas[name].obs.pixie, normalize='columns'),
           ax=ax2.flatten()[1])
ax2.flatten()[1].set_title('Norm within pixie envs')
fig.savefig(f'Figures/heatmaps_{name}.png', bbox_inches='tight', dpi=300)

# Save AnnData
adatas[name].write_h5ad(f'{name}.h5ad')