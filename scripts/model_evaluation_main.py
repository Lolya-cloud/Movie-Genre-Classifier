import os
import sys

import torch.nn as nn
import torch.optim as opt
import pandas as pd
from tabulate import tabulate
from DataProcessor import DataProcessor
from BERT import BertPretrained
from FlexibleNeuralNetwork import FlexibleNeuralNetwork
from NeuralNetworkTrainer import NeuralNetworkTrainer
from ModelEvaluationMetric import ModelEvaluationMetric

# paths for flexibility
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.abspath(script_path))
unprocessed_data_path = "../data/unprocessed_data/imdb_movies.csv"
processed_data_path = "../data/processed_data/processed_data.csv"
tensors_path = "../data/tensors/bert_tensors.pt"
embeddings_path = "../data/embeddings/bert_embeddings.npy"
embeddings_path_sbert = "../data/embeddings/sbert_embeddings.npy"
embeddings_path_word2vec_google = "../data/embeddings/word2vec_google_embeddings.npy"
bert_metric_location = "../analysis/model_comparison/bert_metrics"
sbert_metric_location = "../analysis/model_comparison/sbert_metrics"
w2v_metric_location = "../analysis/model_comparison/w2v_metrics"
max_data_len = 301  # input length for bert. It was found that the largest sequence of tokens produced
# by bert tokenizer for the dataset is 301.


def test_model_architecture(embeddings_train_test, model_name, input_features,
                            metric_file_location, overviews, labels):
    # create model architecture
    # input_features = 768
    output_features = 19
    layer_sizes = [
        [input_features, 100, output_features],
        [input_features, 500, output_features],
        [input_features, 1000, output_features],
        [input_features, 100, 100, 100, output_features],
        [input_features, 500, 500, 500, output_features],
        [input_features, 1000, 1000, 1000, output_features],
        [input_features, 100, 100, 100, 100, 100, output_features],
        [input_features, 500, 500, 500, 500, 500, output_features],
        [input_features, 1000, 1000, 1000, 1000, 1000, output_features]
                   ]

    dropout_rates = [
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ]
    hidden_layer_activation_functions = [nn.ReLU(), nn.PReLU(), nn.Tanh()]
    output_layer_activation = nn.Sigmoid()
    metrics_df = pd.DataFrame()

    for i, layer_size in enumerate(layer_sizes):
        for hidden_layer_activation in hidden_layer_activation_functions:
            # initialize NN
            model = FlexibleNeuralNetwork(layer_sizes=layer_size, dropout_rates=dropout_rates[i],
                                          hidden_layer_activation=hidden_layer_activation,
                                          output_layer_activation=output_layer_activation)
            # set training parameters
            learning_rate = 0.001
            epochs = 100
            training_batch = 100
            test_split = 0.2
            random_state = 10
            loss_function = nn.BCELoss()
            optimizer = opt.Adam(model.parameters(), lr=learning_rate)
            model_description = f"model_name: {model_name}, model_architecture: " \
                                f"[layer_sizes: {layer_size}, dropout_layers: {dropout_rates[i]}, " \
                                f"hidden layer activation: {type(hidden_layer_activation).__name__}, " \
                                f"final_layer_activation: {type(output_layer_activation).__name__}], " \
                                f"optimizer: {type(optimizer).__name__}, " \
                                f"loss_function: {type(loss_function).__name__} " \
                                f"learning_rate: {learning_rate}, epochs: {epochs}, test_split_ratio: {test_split}, " \
                                f"random split state: {random_state}"

            # initialize trainer
            trainer = NeuralNetworkTrainer(embeddings=embeddings_train_test, labels=labels, raw_values=overviews,
                                           model=model, loss_function=loss_function, optimizer=optimizer,
                                           batch_size=training_batch, test_split_ratio=test_split,
                                           split_random_state=random_state)

            # train the model:
            trainer.train(epochs)

            # evaluate the model
            decission_threshold = 0.3
            metric, comparison_data = trainer.evaluate(decission_threshold=decission_threshold,
                                                       model_name=model_name, model_description=model_description)
            metrics_dict = metric.get_metrics()
            new_row = pd.DataFrame([metrics_dict])
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    path = os.path.join(script_dir, metric_file_location)
    metrics_df.to_csv(path, index=False)


# before training.
processor = DataProcessor(script_path, unprocessed_data_path)
input_ids, attention_masks, labels = processor.process_data_for_BERT(tensors_path, max_data_len)
tensor_dicts = [{"input_ids": id_tensor, "attention_mask": mask_tensor}
                for id_tensor, mask_tensor in zip(input_ids, attention_masks)]
overviews = processor.dataset['overview']
bert = BertPretrained(script_path)
# get embeddings for each text in the dataset
embeddings = bert.get_embeddings(tensor_dicts, embeddings_path)
# embeddings, labels = processor.process_word2vec_google(embeddings_path_word2vec_google)

model_name = "Bert with cls and NN"
test_model_architecture(embeddings, model_name, 768, bert_metric_location, overviews, labels)
print("bert done, starting sbert")

model_name = "SBERT with NN"
embeddings, labels = processor.process_sbert(embeddings_path_sbert)
test_model_architecture(embeddings, model_name, 768, sbert_metric_location, overviews, labels)
print("sbert done, starting w2v")

model_name = "Word2vec pretrained by google with NN"
embeddings, labels = processor.process_word2vec_google(embeddings_path_word2vec_google)
test_model_architecture(embeddings, model_name, 300, w2v_metric_location, overviews, labels)
