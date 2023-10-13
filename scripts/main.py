import os
from DataProcessor import DataProcessor

script_path = os.path.abspath(__file__)
unprocessed_data_path = "../data/unprocessed_data/imdb_movies.csv"
processed_data_path = "../data/processed_data/processed_data.csv"
max_data_len = 200

processor = DataProcessor(script_path, unprocessed_data_path)
# saving and loading dataset. note that tokens for bert are in list format, not tensors, so conversion is required
# before training.
dataset = processor.process_data_for_BERT(processed_data_path, max_data_len)
print(dataset.head())
# convert lists to tensors before training
processor.convert_lists_to_tensors(dataset)


