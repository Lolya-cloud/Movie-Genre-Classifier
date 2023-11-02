import os
import sys
import numpy as np
from scripts.DataProcessor import DataProcessor
from scripts.BERT import BertPretrained
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
embeddings, labels = processor.process_word2vec_google(os.path.join(script_dir, embeddings_path_word2vec_google))
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

embeddings_sum_for_each_label = defaultdict(lambda: np.zeros(len(embeddings[0]), dtype=float))
counter_for_each_unique_label = defaultdict(int)
for i, labels_set in enumerate(labels):
    for label in labels_set:
        embeddings_sum_for_each_label[label] += bert_embeddings[i]
        counter_for_each_unique_label[label] += 1

centroids = {}
for label in embeddings_sum_for_each_label:
    if label not in counter_for_each_unique_label:
        raise ValueError(f"The label: {label} doesn't exist in counter_for_each_unique_label")
    centroids[label] = embeddings_sum_for_each_label[label]/counter_for_each_unique_label[label]

cenroid_matrix = np.array(list(centroids.values()))
similarity_matrix = cosine_similarity(cenroid_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=label_categories, columns=label_categories)
print(similarity_df.head(10))

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, cbar=True)
plt.title('Similarity Matrix', fontsize=20)
plt.tight_layout()

plt.show()
