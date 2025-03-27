import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import os
import networkx as nx
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import itertools

from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.data import BiasedRandomWalk
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification

from stellargraph import datasets
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
roi_names= pd.Series(roi_names).astype('category')
list_of_lists = [[roi for x in range(graph_size)] for roi, graph_size in zip(roi_names, [g.number_of_nodes() for g in G_list])]
rois = list(itertools.chain.from_iterable(list_of_lists))

# Rename 'label' label, as StellarGraph sees it as different node types
for node, data in G_nx.nodes(data=True):
    if "label" in data:
        data["ecm_env"] = data.pop("label")

# Load node data into a pandas dataframe
node_data = pd.DataFrame.from_dict(dict(G_nx.nodes(data=True)), orient="index")
node_data.loc[:,'pixie'] = node_data['ecm_env'].astype(int).map(pixie_envs) 
node_data.loc[:,'roi'] = rois

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

# Add area
node_features.loc[:, 'area'] = node_data.loc[:,'area_norm'] 

# Convert to StellarGraph
G= StellarGraph.from_networkx(G_nx, node_features=node_features)

# -----------------------------------------------
# Set up Node2Vev
# -----------------------------------------------

walker = BiasedRandomWalk(
    G,
    n=10, # Number of walks per node
    length=5, # Length of each walk
    p=0.5,  # defines probability, 1/p, of returning to source node
    q=2.0,  # defines probability, 1/q, for moving to a node away from the source node
)

unsupervised_samples = UnsupervisedSampler(G, nodes=list(G.nodes()), walker=walker)

batch_size = 50
epochs = 2
generator = Node2VecLinkGenerator(G, batch_size)

emb_size = 128
node2vec = Node2Vec(emb_size, generator=generator)
x_inp, x_out = node2vec.in_out_tensors()

prediction = link_classification(
    output_dim=1, 
    output_act="sigmoid", 
    edge_embedding_method="dot"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

# -----------------------------------------------
# Fit Node2Vev
# -----------------------------------------------

history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=8,
    shuffle=True,
)


# -----------------------------------------------
# Retrieve and save embeddings as Numpy array file
# -----------------------------------------------

x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

node_gen = Node2VecNodeGenerator(G, batch_size).flow(node_data.index)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

np.save('node_embeddings_node2vec.npy', node_embeddings)