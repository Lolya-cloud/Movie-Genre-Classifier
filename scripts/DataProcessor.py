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

    def extract_features_dataset(self, dataset):
        return dataset[['overview', 'genre']]

    def process_data_for_BERT(self, relative_output_path, max_len):
        """
        Main entry axis. Processes data, cleans it and prepares for training. If the path already exists,
        attempts to load the dataset instead. Prepares data for BERT model specifically.
        :param relative_output_path: relative path to the output file, where processed data will be saved.
        :return: processed dataset - dataset tokenized using bert tokenizer and ready for feeding to the BERT monster
        """
        # check if file already exists
        if os.path.exists(relative_output_path):
            return self.load_data(relative_output_path)

        # basic cleaning
        dataset = self.load_data(self.unprocessed_data_path)
        dataset = self.basic_process_tokenize(dataset)
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset['bert_encoded'] = dataset['overview'].apply(lambda x: self.encode_for_bert(x, max_len,
                                                                                                     bert_tokenizer))
        self.save_data(dataset, relative_output_path)
        return dataset

    def convert_lists_to_tensors(self, dataset):
        dataset['bert_encoded'] = dataset['bert_encoded'].apply(lambda x: {key: torch.tensor(value) for key, value in x.items()})
        return dataset

    def basic_process_tokenize(self, dataset):
        dataset = self.extract_features_dataset(dataset)
        return dataset

    def basic_tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", '', text)
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in stopwords.words('english')]

    def encode_for_bert(self, tokenized_text, max_len, bert_tokenizer):
        tokens = bert_tokenizer.encode_plus(tokenized_text, add_special_tokens=True, max_length=max_len,
                                       padding='max_length', truncation=True, return_tensors='pt')
        # return dictionary of tensors converted to lists
        return {key: val[0].tolist() for key, val in tokens.items()}