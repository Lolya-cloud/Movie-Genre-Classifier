from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, hamming_loss


class ModelEvaluationMetric:
    def __init__(self, labels, y_pred, model_name, model_description, used_decission_threshold,
                 best_threshold_f1_score, best_decission_threshold=None):
        self.labels = labels
        self.y_pred = y_pred
        self.model_name = model_name
        self.model_description = model_description
        self.accuracy = accuracy_score(y_true=labels, y_pred=y_pred)
        self.precision = precision_score(y_true=labels, y_pred=y_pred, average='micro')
        self.recall = recall_score(y_true=labels, y_pred=y_pred, average='micro')
        self.f1 = f1_score(y_true=labels, y_pred=y_pred, average='micro')
        self.best_decission_threshold = best_decission_threshold
        self.used_decission_threshold = used_decission_threshold
        self.best_threshold_f1_score = best_threshold_f1_score

        # note: accuracy isn't a good metric for multi-label classification, as it does not consider label subsets.
        # hence, we also use jaccard score and hamming loss.
        self.hamming_accuracy = 1 - hamming_loss(y_true=labels, y_pred=y_pred)
        self.jaccard_score_ = jaccard_score(y_true=labels, y_pred=y_pred, average='micro')

    def display_metrics(self):
        print(f'Accuracy: {self.accuracy}, Hamming Accuracy: {self.hamming_accuracy}, '
              f'Jaccard Score: {self.jaccard_score_}, Precision: {self.precision}, '
              f'Recall: {self.recall}, F1 Score: {self.f1}, '
              f'Optimal decission threshold based on F1-curve: {self.best_decission_threshold}')

    def get_metrics(self):
        metrics = {
            'model_name': self.model_name,
            'model_description': self.model_description,
            'accuracy': self.accuracy,
            'hamming_accuracy': self.hamming_accuracy,
            'jaccard_score': self.jaccard_score_,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1,
            'used_decission_threshold': self.used_decission_threshold,
            'best_decission_threshold': self.best_decission_threshold,
            'best_threshold_f1_score': self.best_threshold_f1_score
        }
        return metrics
