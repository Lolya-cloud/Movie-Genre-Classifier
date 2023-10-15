import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
import torch


class DataProcessor:
    def __init__(self, script_path, unprocessed_data_path):
        self.script_dir = os.path.dirname(os.path.abspath(script_path))
        self.unprocessed_data_path = os.path.join(self.script_dir, unprocessed_data_path)

    def load_data(self, path):
        return pd.read_csv(path)

    def save_data(self, output_data, output_data_path):
        absolute_path = os.path.join(self.script_dir, output_data_path)
        output_data.to_csv(absolute_path, index=False)

    def save_tensors(self, tensors, tensors_path):
        absolute_path = os.path.join(self.script_dir, tensors_path)
        torch.save(tensors, absolute_path)

    def load_tensors(self, tensors_path):
        absolute_path = os.path.join(self.script_dir, tensors_path)
        return torch.load(absolute_path)

    def extract_features_dataset(self, dataset):
        return dataset[['overview', 'genre']]

    def process_data_for_BERT(self, relative_tensor_output_path, max_len):
        # Check if tensor file already exists
        if os.path.exists(os.path.join(self.script_dir, relative_tensor_output_path)):
            print("Tensors found, loading")
            bert_encoded_tensors = self.load_tensors(relative_tensor_output_path)
            return bert_encoded_tensors
        print("Tensors not found, generating")
        dataset = self.load_data(self.unprocessed_data_path)
        dataset = self.basic_process(dataset)
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        bert_encoded = dataset['overview'].apply(
            lambda x: self.encode_for_bert(x, max_len, bert_tokenizer)
        ).tolist()

        input_ids = torch.stack([item[0] for item in bert_encoded])
        attention_masks = torch.stack([item[1] for item in bert_encoded])

        self.save_tensors((input_ids, attention_masks), relative_tensor_output_path)

        return input_ids, attention_masks

    def basic_process(self, dataset):
        dataset = self.extract_features_dataset(dataset)
        return dataset

    def encode_for_bert(self, tokenized_text, max_len, bert_tokenizer):
        tokens = bert_tokenizer.encode_plus(
            tokenized_text, add_special_tokens=True, max_length=max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        # Return both input_ids and attention_mask tensors
        return tokens['input_ids'][0], tokens['attention_mask'][0]
