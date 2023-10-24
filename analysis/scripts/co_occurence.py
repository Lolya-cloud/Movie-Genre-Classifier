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
unprocessed_data_path = "../../data/unprocessed_data/imdb_movies.csv"
processed_data_path = "../data/processed_data/processed_data.csv"
tensors_path = "../../data/tensors/bert_tensors.pt"
embeddings_path = "../../data/embeddings/bert_embeddings.npy"
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

co_occurrence_matrix = pd.DataFrame(0, index=label_categories, columns=label_categories, dtype=np.int64)

# CHATGPT CHATGPT CHATGPT
# Step 1: Calculate the occurrence of each genre.
genre_counts = {genre: 0 for genre in label_categories}
for movie_genres in splitted_labels:
    for genre in movie_genres:
        genre_counts[genre] += 1

# Create a DataFrame for the co-occurrence matrix.
co_occurrence_matrix = pd.DataFrame(0, index=label_categories, columns=label_categories, dtype=np.float64)

# Step 2: Calculate normalized co-occurrence.
for movie_genres in splitted_labels:
    for i in range(len(movie_genres)):
        for j in range(i + 1, len(movie_genres)):  # Avoid double counting
            genre_i = movie_genres[i]
            genre_j = movie_genres[j]

            # Calculate normalization factor.
            normalization_factor = np.sqrt(genre_counts[genre_i] * genre_counts[genre_j])

            # Update co-occurrence matrix.
            co_occurrence_matrix.loc[genre_i, genre_j] += 1 / normalization_factor
            co_occurrence_matrix.loc[genre_j, genre_i] += 1 / normalization_factor

# Now, co_occurrence_matrix contains the relative co-occurrence frequencies.

# Visualize the matrix
plt.figure(figsize=(12, 9))
sns.heatmap(co_occurrence_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Normalized Genre Co-occurrence')
plt.show()
print(len(label_categories))

# Count the frequency of each label count (e.g., how many movies have exactly 1 label, 2 labels, etc.).
label_count_freq = {}
num_labels_per_movie = [len(labels) for labels in splitted_labels]

for count in num_labels_per_movie:
    if count in label_count_freq:
        label_count_freq[count] += 1
    else:
        label_count_freq[count] = 1

# Data for plotting.
counts = list(label_count_freq.keys())
frequencies = list(label_count_freq.values())

# Creating the histogram.
plt.figure(figsize=(10, 6))
plt.bar(counts, frequencies, color='blue', alpha=0.7)
plt.xlabel('Number of Genres per Movie')
plt.ylabel('Number of Movies')
plt.title('Frequency of Number of Genres per Movie')
plt.xticks(counts)  # Ensure all categories are shown on x-axis.
plt.grid(axis='y', alpha=0.5)

# Adding the text labels on each bar.
for i in range(len(counts)):
    plt.text(i + min(counts) - 0.25,  # X location of text (with slight adjustment to center).
             frequencies[i] + (max(frequencies) * 0.01),  # Y location of text.
             str(frequencies[i]),  # The text (frequency count).
             color='blue')

plt.show()

overview_lengths = [len(overview.split()) for overview in overviews]  # List of lengths of each overview

# You might want to see the distribution of overview lengths.
plt.figure(figsize=(10, 6))
plt.hist(overview_lengths, bins=50, color='blue', alpha=0.7)  # Customize the number of bins as you see fit
plt.xlabel('Length of Overview')
plt.ylabel('Number of Movies')
plt.title('Distribution of Movie Overview Lengths')
plt.grid(axis='y', alpha=0.5)
plt.show()


# Group embeddings based on the length of the overviews
length_to_embeddings = defaultdict(list)
for length, embedding in zip(overview_lengths, embeddings):
    length_to_embeddings[length].append(embedding)

# For each group, calculate the average similarity between embeddings
group_similarities = {}
for length, group_embeddings in length_to_embeddings.items():
    if len(group_embeddings) > 1:  # Ensure there are at least two embeddings to compare
        group_sim_matrix = cosine_similarity(group_embeddings)
        avg_similarity = np.mean(group_sim_matrix)
        group_similarities[length] = avg_similarity

# Plotting the average similarity per group based on the text length
lengths = list(group_similarities.keys())
similarities = list(group_similarities.values())

plt.figure(figsize=(10, 6))
plt.scatter(lengths, similarities, color='blue', alpha=0.7)
plt.xlabel('Length of Overview')
plt.ylabel('Average Embedding Similarity')
plt.title('Embedding Similarity by Overview Length')
plt.grid(axis='both', alpha=0.5)
plt.show()

