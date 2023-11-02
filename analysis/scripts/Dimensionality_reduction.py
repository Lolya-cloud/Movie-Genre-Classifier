import os
import sys
import numpy as np
from scripts.DataProcessor import DataProcessor
from scripts.BERT import BertPretrained
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.abspath(script_path))
unprocessed_data_path = "../../data/unprocessed_data/imdb_movies.csv"
processed_data_path = "../data/processed_data/processed_data.csv"
tensors_path = "../../data/tensors/bert_tensors.pt"
embeddings_path = "../../data/embeddings/bert_embeddings.npy"
embeddings_path_sbert = "../../data/embeddings/sbert_embeddings.npy"
embeddings_path_word2vec_google = "../../data/embeddings/word2vec_google_embeddings.npy"
max_data_len = 301
processor = DataProcessor(script_path, unprocessed_data_path)
# before training.
input_ids, attention_masks, labels = processor.process_data_for_BERT(tensors_path, max_data_len)
tensor_dicts = [{"input_ids": id_tensor, "attention_mask": mask_tensor} for id_tensor, mask_tensor in zip(input_ids, attention_masks)]

bert = BertPretrained(script_path)
# get embeddings for each text in the dataset
embeddings = bert.get_embeddings(tensor_dicts, embeddings_path)
embeddings, labels = processor.process_sbert(os.path.join(script_dir, embeddings_path_sbert))

max_data_len = 301
processor = DataProcessor(script_path, unprocessed_data_path)
# before training.
input_ids, attention_masks, labels = processor.process_data_for_BERT(tensors_path, max_data_len)
tensor_dicts = [{"input_ids": id_tensor, "attention_mask": mask_tensor}
                for id_tensor, mask_tensor in zip(input_ids, attention_masks)]

bert = BertPretrained(script_path)
# get embeddings for each text in the dataset
embeddings = bert.get_embeddings(tensor_dicts, embeddings_path)
overviews = processor.dataset['overview']
# prepare labels
splitted_labels = [label.split(',') for label in labels]

flattened_labels = []
for x in splitted_labels:
    for y in x:
        flattened_labels.append(y)

# find unique set of labels
label_categories = np.unique(np.array(flattened_labels))

# all labels in an array of arrays
labels = splitted_labels
# all embeddings
bert_embeddings = embeddings

tsne = TSNE(n_components=3, random_state=42)  # fixating random state
transformed_embeddings = tsne.fit_transform(embeddings)
unique_genres = label_categories

fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111)
colors = mpl.colormaps['jet'](np.linspace(0, 1, len(unique_genres)))
genre_to_color = {genre: color for genre, color in zip(unique_genres, colors)}
for i, embedding in enumerate(transformed_embeddings):
    first_genre = labels[i][0]
    ax.scatter(embedding[0], embedding[1], color=genre_to_color[first_genre])

ax.set_title('2D visualization of embeddings with t-SNE')
plt.show()
