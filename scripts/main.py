import os
from DataProcessor import DataProcessor
from BERT import BertPretrained
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch.nn as nn
import torch.optim as opt


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

label_categories = np.unique(np.array(flattened_labels))
print(label_categories)

mlb = MultiLabelBinarizer(classes=label_categories)
encoded = mlb.fit_transform(splitted_labels)
print(encoded)
