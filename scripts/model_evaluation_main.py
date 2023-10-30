import os
import torch.nn as nn
import torch.optim as opt
import pandas as pd
from DataProcessor import DataProcessor
from BERT import BertPretrained
from FlexibleNeuralNetwork import FlexibleNeuralNetwork
from NeuralNetworkTrainer import NeuralNetworkTrainer
from ModelEvaluationMetric import ModelEvaluationMetric

# paths for flexibility
script_path = os.path.abspath(__file__)
unprocessed_data_path = "../data/unprocessed_data/imdb_movies.csv"
processed_data_path = "../data/processed_data/processed_data.csv"
tensors_path = "../data/tensors/bert_tensors.pt"
embeddings_path = "../data/embeddings/bert_embeddings.npy"
max_data_len = 301  # input length for bert. It was found that the largest sequence of tokens produced
# by bert tokenizer for the dataset is 301.

# before training.
processor = DataProcessor(script_path, unprocessed_data_path)
input_ids, attention_masks, labels = processor.process_data_for_BERT(tensors_path, max_data_len)
tensor_dicts = [{"input_ids": id_tensor, "attention_mask": mask_tensor}
                for id_tensor, mask_tensor in zip(input_ids, attention_masks)]

bert = BertPretrained(script_path)
# get embeddings for each text in the dataset
embeddings = bert.get_embeddings(tensor_dicts, embeddings_path)
overviews = processor.dataset['overview']

# create model architecture
input_features = 768
output_features = 19
layer_sizes = [input_features, 384, 192, 96, 48, 24, output_features]
dropout_rates = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
hidden_layer_activation = nn.ReLU()
output_layer_activation = nn.Sigmoid()

# initialize NN
model = FlexibleNeuralNetwork(layer_sizes=layer_sizes, dropout_rates=dropout_rates,
                              hidden_layer_activation=hidden_layer_activation,
                              output_layer_activation=output_layer_activation)

# set training parameters
learning_rate = 0.001
epochs = 100
training_batch = 50
test_split = 0.2
random_state = 10
loss_function = nn.BCELoss()
optimizer = opt.Adam(model.parameters(), lr=learning_rate)
model_name = "Simple NN with Bert CLS"
model_description = f"model_name: {model_name}, model_architecture: " \
                    f"[layer_sizes: {layer_sizes}, dropout_layers: {dropout_rates}, " \
                    f"hidden layer activation: {type(hidden_layer_activation).__name__}, " \
                    f"final_layer_activation: {type(output_layer_activation).__name__}], " \
                    f"optimizer: {type(optimizer).__name__}, " \
                    f"loss_function: {type(loss_function).__name__}" \
                    f"learning_rate: {learning_rate}, epochs: {epochs}, test_split_ratio: {test_split}, " \
                    f"random split state: {random_state}"

# initialize trainer
trainer = NeuralNetworkTrainer(embeddings=embeddings, labels=labels, raw_values=overviews,
                               model=model, loss_function=loss_function, optimizer=optimizer,
                               batch_size=training_batch, test_split_ratio=test_split,
                               split_random_state=random_state)

# train the model:
trainer.train(epochs)

# evaluate the model
decission_threshold = .3
metric, comparison_data = trainer.evaluate(decission_threshold=decission_threshold,
                                           model_name=model_name, model_description=model_description)

metric.display_metrics()
print(metric.get_metrics())
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.head())  # Print the first few rows to check
comparison_df.to_csv('label_comparison.csv', index=False)