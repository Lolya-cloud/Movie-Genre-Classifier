import os
import sys

from DataProcessor import DataProcessor
from BERT import BertPretrained
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch.optim as opt
import torch
from torch.utils.data import TensorDataset, DataLoader
from NN import NN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, hamming_loss

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.abspath(script_path))
unprocessed_data_path = "../data/unprocessed_data/imdb_movies.csv"
processed_data_path = "../data/processed_data/processed_data.csv"
tensors_path = "../data/tensors/bert_tensors.pt"
embeddings_path = "../data/embeddings/bert_embeddings.npy"
embeddings_path_tf_idf = "../data/embeddings/tf-idf.npy"
max_data_len = 301  # input length for bert. It was found that the largest sequence of tokens produced
# by bert tokenizer for the dataset is 301.

processor = DataProcessor(script_path, unprocessed_data_path)
# before training.
input_ids, attention_masks, labels = processor.process_data_for_BERT(tensors_path, max_data_len)
tensor_dicts = [{"input_ids": id_tensor, "attention_mask": mask_tensor}
                for id_tensor, mask_tensor in zip(input_ids, attention_masks)]

test1, test2 = processor.process_data_tf_idf(embeddings_path_tf_idf)
print(f"shape embeddings tf-idf: {test1.shape}")
print(test2.shape)
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

print(len(embeddings))
# find unique set of labels
label_categories = np.unique(np.array(flattened_labels))

# use binarizer to create a  tupple matrix of size (num of labels) * (unique labels),
# where each cell indicates presence
# or absence of the particular label (either 1 or 0)
mlb = MultiLabelBinarizer(classes=label_categories)
labels_matrix = mlb.fit_transform(splitted_labels)
print(labels_matrix)

# random state allows to always get the same split (for testing purposes)
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_matrix, test_size=0.2, random_state=10)
# get overviews as well
_, overviews_test = train_test_split(overviews, test_size=0.2, random_state=10)

print(np.sum(y_train, axis=0))
print(np.sum(y_test, axis=0))

if torch.cuda.is_available():
    print("GPU found, will be used for training")
    device = torch.device("cuda")
else:
    print("Warning: GPU not found, using CPU for training. Expected training time: FOREVER")
    device = torch.device('cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# training in batches of 50
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# do not shuffle for possibility of feature comparison.
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
print(f'input len: {X_train.shape[1]}, output len: {len(label_categories)}')
# create the model
model = NN(input_len=X_train.shape[1], output_len=len(label_categories))
print(f'input len: {X_train.shape[1]}, output len: {len(label_categories)}')
model.to(device)
loss_func = nn.BCELoss()
optimizer = opt.Adam(model.parameters(), lr=0.001) # perfect, slightly better than the rest.
# different optimizers tested. also, different loss functions were tested as well.
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01) / better
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) / ok
#optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0) / ok


# Training. Note: heavily inspired by documentation and ChatGPT 4.
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch: {epoch + 1}, Loss: {total_loss/len(train_loader)}')

# testing
model.eval()
predictions = []
labels = []
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())
        labels.append(targets.cpu().numpy())

labels = np.concatenate(labels, axis=0)
predictions = np.concatenate(predictions, axis=0)

# chatgpt, to be removed
from sklearn.metrics import precision_recall_curve

# Assuming y_test is your true labels data
precision, recall, thresholds = precision_recall_curve(y_test.ravel(), predictions.ravel())
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = np.argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

y_pred = (predictions > 0.33).astype(int)
# chatgpt over


accuracy = accuracy_score(y_true=labels, y_pred=y_pred)
precision = precision_score(y_true=labels, y_pred=y_pred, average='micro')
recall = recall_score(y_true=labels, y_pred=y_pred, average='micro')
f1 = f1_score(y_true=labels, y_pred=y_pred, average='micro')

# note: accuracy isn't a good metric for multi-label classification, as it does not consider label subsets.
# hence, we also use jaccard score and hamming loss.
hamming_loss = hamming_loss(y_true=labels, y_pred=y_pred)
hamming_acc = 1 - hamming_loss
jaccard_score = jaccard_score(y_true=labels, y_pred=y_pred, average='micro')
print(f'Accuracy: {accuracy}, Hamming Accuracy: {hamming_acc}, '
      f'Jaccard Score: {jaccard_score}, Precision: {precision}, '
      f'Recall: {recall}, F1 Score: {f1}')

# print inversed
true_labels_text = mlb.inverse_transform(labels)
predicted_labels_text = mlb.inverse_transform(y_pred)  # Assuming you've rounded predictions to 0 or 1


# chatgpt onwards, to be removed
import pandas as pd

comparison_data = {
    "True Labels": [', '.join(labels) for labels in true_labels_text],
    "Predicted Labels": [', '.join(labels) for labels in predicted_labels_text],
    "Original Overviews": overviews_test
}
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.head())  # Print the first few rows to check
comparison_df.to_csv('label_comparison.csv', index=False)


# firstly, let's determine mostly misclassified labels. GENERATED by CHATGPT
from collections import Counter

# Extract misclassified labels
misclassified = []
for true, pred in zip(true_labels_text, predicted_labels_text):
    if true != pred:
        misclassified.extend(true)  # Adding true labels that were misclassified

# Count the occurrences of each misclassified label
misclassified_counts = Counter(misclassified)

# Most common misclassified labels
most_misclassified = misclassified_counts.most_common()  # You can specify a number inside most_common() to get the top N

print("Most misclassified labels:")
for label, count in most_misclassified:
    print(f"{label}: {count}")

import matplotlib.pyplot as plt

# Count the occurrences of each unique label in the original dataset
original_label_counts = Counter(flattened_labels)

# Data for plotting
labels, frequencies = zip(*original_label_counts.items())

# Creating histogram
plt.figure(figsize=(15, 7))
plt.bar(labels, frequencies)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.xticks(rotation=90)  # Rotates X-Axis Labels
plt.title('Distribution of labels in the dataset')
plt.tight_layout()  # To ensure labels are neatly laid out

# Display the histogram
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(15, 14))  # 2 rows, 1 column

# Histogram of original labels
axs[0].bar(labels, frequencies)
axs[0].set_title('Original Labels Distribution')
axs[0].set_xlabel('Labels')
axs[0].set_ylabel('Frequency')
axs[0].tick_params(axis='x', rotation=90)  # Rotates X-Axis Labels

# Bar chart for most misclassified labels
mis_labels, mis_frequencies = zip(*most_misclassified)
axs[1].bar(mis_labels, mis_frequencies, color='red')
axs[1].set_title('Most Misclassified Labels')
axs[1].set_xlabel('Labels')
axs[1].set_ylabel('Count')
axs[1].tick_params(axis='x', rotation=90)  # Rotates X-Axis Labels

plt.tight_layout()
plt.show()

from collections import defaultdict

# Prepare structures for counting
label_counts = defaultdict(int)
misclassified_counts = defaultdict(int)

# Count each label and each misclassified label
for true, pred in zip(true_labels_text, predicted_labels_text):
    true_set = set(true)
    pred_set = set(pred)

    # Update label counts
    for label in true_set:
        label_counts[label] += 1

    # Find misclassified labels and update counts
    if true_set != pred_set:
        for label in true_set.difference(pred_set):
            misclassified_counts[label] += 1

# Calculate misclassification rate for each label
misclassification_rates = {label: misclassified_counts[label] / label_counts[label] for label in label_counts}

# It might be useful also to have the number of occurrences for each label for later analysis
label_occurrences = {label: count for label, count in label_counts.items()}

import matplotlib.pyplot as plt

# Extract data for plotting
labels, rates = zip(*misclassification_rates.items())
_, occurrences = zip(*label_occurrences.items())

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(occurrences, rates)

# Annotate each point with the corresponding label name
for i, label in enumerate(labels):
    plt.annotate(label, (occurrences[i], rates[i]), fontsize=9, alpha=0.7)

plt.title('Relation between Label Occurrence and Misclassification Rate')
plt.xlabel('Number of occurrences in the dataset')
plt.ylabel('Misclassification rate')
plt.xscale('log')  # Using log scale due to wide range of values
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
