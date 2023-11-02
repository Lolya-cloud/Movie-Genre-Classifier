import os
import pandas as pd


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(os.path.abspath(script_path))
bert_metric_location = "../analysis/model_comparison/bert_metrics"
sbert_metric_location = "../analysis/model_comparison/sbert_metrics"
w2v_metric_location = "../analysis/model_comparison/w2v_metrics"
merge_location = "../analysis/model_comparison/merged_metrics"

bert_df = pd.read_csv(os.path.join(script_dir, bert_metric_location))
sbert_df = pd.read_csv(os.path.join(script_dir, sbert_metric_location))
w2v_df = pd.read_csv(os.path.join(script_dir, w2v_metric_location))

total_df = pd.concat([bert_df, sbert_df, w2v_df], ignore_index=True)
total_df.to_csv(os.path.join(script_dir, merge_location))
