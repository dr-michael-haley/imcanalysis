# -----------------------------------------------
# Adapted from:
# "Node representation learning with GraphSAGE and UnsupervisedSampler"
# https://stellargraph.readthedocs.io/en/stable/demos/embeddings/graphsage-unsupervised-sampler-embeddings.html
# -----------------------------------------------

import networkx as nx
import pandas as pd
import numpy as np
import os
import random
import itertools

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow as tf

from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score

from stellargraph import globalvar

from IPython.display import display, HTML

if tf.config.list_physical_devices('GPU'):
    print('GPU available!')
else:
    print('NO GPU available?')

print(tf.config.list_physical_devices('GPU'))

# -----------------------------------------------
# Load NetworkX networks for matrix environments
# -----------------------------------------------

# Get list of graphs and make into one
G_list = [nx.read_gexf(f'Matrix_networkx_graphs/{f}') for f in os.listdir('Matrix_networkx_graphs')]
G_nx = nx.disjoint_union_all(G_list)

# Annotate with ROI names
roi_names = os.listdir('Matrix_networkx_graphs')
roi_names = [x.replace('_pixel_mask.gexf', '') for x in roi_names]
roi_names = pd.Series(roi_names).astype('category')
list_of_lists = [[roi for x in range(graph_size)] for roi, graph_size in
                 zip(roi_names, [g.number_of_nodes() for g in G_list])]
rois = list(itertools.chain.from_iterable(list_of_lists))

# Rename 'label' label, as StellarGraph sees it as different node types
for node, data in G_nx.nodes(data=True):
    if "label" in data:
        data["ecm_env"] = data.pop("label")

# Load node data into a pandas dataframe
node_data = pd.DataFrame.from_dict(dict(G_nx.nodes(data=True)), orient="index")
node_data.loc[:, 'pixie'] = node_data['ecm_env'].astype(int).map(pixie_envs)
node_data.loc[:, 'roi'] = rois

# This is used downstream
subjects = node_data['pixie']

# Calculate area_proportion of nodes (rather than just raw area)
for r in node_data.roi.unique().tolist():
    # specific roi
    node_data_roi = node_data.loc[node_data.roi == r, :]

    total_area = node_data_roi.area.sum()

    node_data.loc[node_data.roi == r, 'area_norm'] = node_data_roi['area'].div(total_area)

# One hot encoding of pixie envs
node_features = pd.get_dummies(node_data['pixie'])

##### Skipping adding area
# Add area
# node_features.loc[:, 'area'] = node_data.loc[:, 'area_norm']

# Convert to StellarGraph
G = StellarGraph.from_networkx(G_nx, node_features=node_features)

# -----------------------------------------------
# Set up GraphSage
# -----------------------------------------------

# Specify the  parameter values: root nodes (all, in this case), the number of walks to take per node, the length of each walk, and random seed.**
nodes = list(G.nodes())
number_of_walks = 10
length = 5

unsupervised_samples = UnsupervisedSampler(
    G, nodes=nodes, length=length, number_of_walks=number_of_walks
)

batch_size = 50
epochs = 5
num_samples = [10, 5]

generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
train_gen = generator.flow(unsupervised_samples)

layer_sizes = [50, 50]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
)

x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

# -----------------------------------------------
# Fit GraphSage
# -----------------------------------------------

history = model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=8,
    shuffle=True,
)

# -----------------------------------------------
# Retrieve and save embeddings as Numpy array file
# -----------------------------------------------

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

node_ids = subjects.index
node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(node_ids)

node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

np.save('node_embedding_graphsage.npy', node_embeddings)