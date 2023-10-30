import numpy as np
from DataProcessor import DataProcessor
from BERT import BertPretrained
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as opt
import torch
from torch.utils.data import TensorDataset, DataLoader
from NN import NN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, hamming_loss
from sklearn.metrics import precision_recall_curve
from ModelEvaluationMetric import ModelEvaluationMetric


class NeuralNetworkTrainer():
    def __init__(self, embeddings, labels, raw_values, model, loss_function, optimizer,
                 batch_size, test_split_ratio, split_random_state):
        # prepare labels
        splitted_labels = [label.split(',') for label in labels]

        flattened_labels = []
        for x in splitted_labels:
            for y in x:
                flattened_labels.append(y)

        # find unique set of labels
        self.unique_labels = np.unique(np.array(flattened_labels))
        self.labels = splitted_labels
        # use binarizer to create a  tupple matrix of size (num of labels) * (unique labels),
        # where each cell indicates presence
        # or absence of the particular label (either 1 or 0)
        self.mlb = MultiLabelBinarizer(classes=self.unique_labels)
        labels_matrix = self.mlb.fit_transform(splitted_labels)
        print(labels_matrix)

        # random state allows to always get the same split (for testing purposes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(embeddings, labels_matrix,
                                                                                test_size=test_split_ratio,
                                                                                random_state=split_random_state)
        # apply the same split to overviews
        _, self.overviews_test = train_test_split(raw_values, test_size=test_split_ratio, random_state=split_random_state)

        print(np.sum(self.y_train, axis=0))
        print(np.sum(self.y_test, axis=0))

        if torch.cuda.is_available():
            print("GPU found, will be used for training")
            self.device = torch.device("cuda")
        else:
            print("Warning: GPU not found, using CPU for training. Expected training time: FOREVER")
            self.device = torch.device('cpu')

        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        # do not shuffle for possibility of feature comparison.
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        self.model = model
        self.model.to(self.device)
        self.loss_func = loss_function
        self.optimizer = optimizer

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(self.train_loader)}')

    def evaluate(self, decission_threshold, model_name, model_description):
        self.model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())

        # evaluation
        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        # chatgpt: how to find best threshold according to F-curve. (ideal balance between precission and recal)
        precision, recall, thresholds = precision_recall_curve(self.y_test.ravel(), predictions.ravel())
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        # chatgpt over

        y_pred = (predictions > decission_threshold).astype(int)

        model_evaluation_metric = ModelEvaluationMetric(labels=labels,
                                                        y_pred=y_pred,
                                                        model_name=model_name,
                                                        model_description=model_description,
                                                        used_decission_threshold=decission_threshold,
                                                        best_decission_threshold=thresholds[ix],
                                                        best_threshold_f1_score=fscore[ix])

        true_labels_text = self.mlb.inverse_transform(labels)
        predicted_labels_text = self.mlb.inverse_transform(y_pred)  # Assuming you've rounded predictions to 0 or 1

        comparison_data = {
            "True Labels": [', '.join(labels) for labels in true_labels_text],
            "Predicted Labels": [', '.join(labels) for labels in predicted_labels_text],
            "Original Overviews": self.overviews_test
        }
        return model_evaluation_metric, comparison_data

