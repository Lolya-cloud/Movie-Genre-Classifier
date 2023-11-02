import numpy
import numpy as np
import pandas as pd
import os
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.utils import simple_preprocess
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import torch


class DataProcessor:
    def __init__(self, script_path, unprocessed_data_path):
        self.script_dir = os.path.dirname(os.path.abspath(script_path))
        self.unprocessed_data_path = os.path.join(self.script_dir, unprocessed_data_path)
        self.dataset = 0

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
        dataset = dataset[['overview', 'genre']]
        return dataset

    def process_data_for_BERT(self, relative_tensor_output_path, max_len):
        dataset = self.load_data(self.unprocessed_data_path)
        dataset, labels = self.basic_process(dataset)
        # save a pointer for later debugging
        self.dataset = dataset
        labels = numpy.array(labels)
        # Check if tensor file already exists
        if os.path.exists(os.path.join(self.script_dir, relative_tensor_output_path)):
            print("Tensors found, loading")
            bert_encoded_tensors = self.load_tensors(relative_tensor_output_path)
            input_ids, att_masks = bert_encoded_tensors
            return input_ids, att_masks, labels
        print("Tensors not found, generating")
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        bert_encoded = dataset['overview'].apply(
            lambda x: self.encode_for_bert(x, max_len, bert_tokenizer)
        ).tolist()

        input_ids = torch.stack([item[0] for item in bert_encoded])
        attention_masks = torch.stack([item[1] for item in bert_encoded])

        self.save_tensors((input_ids, attention_masks), relative_tensor_output_path)

        return input_ids, attention_masks, labels

    def basic_process(self, dataset):
        dataset = self.extract_features_dataset(dataset)
        dataset = dataset.dropna(subset=['genre'])
        labels = dataset['genre'].values.tolist()
        labels = [elem.replace('\xa0', '') for elem in labels]
        return dataset, labels

    def encode_for_bert(self, tokenized_text, max_len, bert_tokenizer):
        tokens = bert_tokenizer.encode_plus(
            tokenized_text, add_special_tokens=True, max_length=max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        # Return both input_ids and attention_mask tensors
        return tokens['input_ids'][0], tokens['attention_mask'][0]

    def process_word2vec_google(self, word2vec_store_path):
        dataset = self.load_data(self.unprocessed_data_path)
        dataset, labels = self.basic_process(dataset)
        abs_path = os.path.join(self.script_dir, word2vec_store_path)
        if os.path.exists(abs_path):
            print("word2vec found, loading.")
            embeddings = numpy.load(abs_path)
            return embeddings, labels
        print('word2vec not found, generating.')
        model = api.load("word2vec-google-news-300")

        tokenized = [simple_preprocess(text) for text in dataset['overview']]

        embeddings = []
        for text in tokenized:
            word_vectors = [model[word] for word in text if word in model]
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(model.vector_size))

        embeddings = np.array(embeddings)
        np.save(abs_path, embeddings)
        return embeddings, labels

    def process_sbert(self, sbert_embeddings_store_path):
        dataset = self.load_data(self.unprocessed_data_path)
        dataset, labels = self.basic_process(dataset)
        abs_path = os.path.join(self.script_dir, sbert_embeddings_store_path)
        if os.path.exists(abs_path):
            print("sbert embeddings found, loading.")
            embeddings = numpy.load(abs_path)
            return embeddings, labels
        print("sbert embeddings not found, generating")
        model = SentenceTransformer('all-mpnet-base-v2')
        embeddings = np.array(model.encode(dataset['overview'].tolist()))
        np.save(abs_path, embeddings)
        return embeddings, labels

