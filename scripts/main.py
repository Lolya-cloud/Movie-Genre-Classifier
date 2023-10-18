import os
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

script_path = os.path.abspath(__file__)
unprocessed_data_path = "../data/unprocessed_data/imdb_movies.csv"
processed_data_path = "../data/processed_data/processed_data.csv"
tensors_path = "../data/tensors/bert_tensors.pt"
embeddings_path = "../data/embeddings/bert_embeddings.npy"
max_data_len = 200 # input length for bert

processor = DataProcessor(script_path, unprocessed_data_path)
# before training.
input_ids, attention_masks, labels = processor.process_data_for_BERT(tensors_path, max_data_len)
tensor_dicts = [{"input_ids": id_tensor, "attention_mask": mask_tensor}
                for id_tensor, mask_tensor in zip(input_ids, attention_masks)]

bert = BertPretrained(script_path)
# get embeddings for each text in the dataset
embeddings = bert.get_embeddings(tensor_dicts, embeddings_path)

# prepare labels
splitted_labels = [label.split(',') for label in labels]
flattened_labels = []
for x in splitted_labels:
    for y in x:
        flattened_labels.append(y)

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

# create the model
model = NN(input_len=X_train.shape[1], output_len=len(label_categories))
model.to(device)
loss_func = nn.BCELoss()
optimizer = opt.Adam(model.parameters(), lr=0.001)

# Training. Note: heavily inspired by documentation and ChatGPT 4.
epochs = 200
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

accuracy = accuracy_score(y_true=labels, y_pred=predictions.round())
precision = precision_score(y_true=labels, y_pred=predictions.round(), average='micro')
recall = recall_score(y_true=labels, y_pred=predictions.round(), average='micro')
f1 = f1_score(y_true=labels, y_pred=predictions.round(), average='micro')

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')