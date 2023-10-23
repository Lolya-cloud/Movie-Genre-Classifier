import torch
from transformers import BertModel
import numpy as np
import os
from tqdm import tqdm


class BertPretrained:
    def __init__(self, script_path):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.to('cuda')
        self.script_dir = os.path.dirname(os.path.abspath(script_path))

    def get_embedding(self, bert_encoded_tensors):
        with torch.no_grad():
            result = self.model(**bert_encoded_tensors)

        # embedding = result.last_hidden_state.mean(dim=1)
        # embedding = result.last_hidden_state.min(dim=1).values
        # mean_pooling = result.last_hidden_state.mean(dim=1)
        # max_pooling = result.last_hidden_state.max(dim=1).values
        # embedding = torch.cat((mean_pooling, max_pooling), dim=1)
        embedding = result.last_hidden_state[:, 0, :] # let's extract the cls token embedding, which
        # represents the entire sequence of words and is great for classification tasks.
        return embedding.cpu().numpy()

    def get_embeddings(self, tensor_dicts, relative_embedding_output_path, batch_size=50):
        if os.path.exists(os.path.join(self.script_dir, relative_embedding_output_path)):
            print("Loading existing embeddings")
            return self.load_embeddings(relative_embedding_output_path)

        print("Embeddings not found, generating")
        embeddings = []
        pbar = tqdm(total=len(tensor_dicts), desc="Generating embeddings")
        # Loop through tensor_dicts in batches
        for i in range(0, len(tensor_dicts), batch_size):
            batch = tensor_dicts[i:i + batch_size]
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_masks = torch.stack([item['attention_mask'] for item in batch])
            input_ids = input_ids.to('cuda')
            attention_masks = attention_masks.to('cuda')

            batch_embeddings = self.get_embedding({
                'input_ids': input_ids,
                'attention_mask': attention_masks
            })

            embeddings.append(batch_embeddings)
            pbar.update(len(batch))
        pbar.close()
        all_embeddings = np.concatenate(embeddings, axis=0)

        print("Embeddings generated, saving")
        self.save_embeddings(all_embeddings, relative_embedding_output_path)
        return all_embeddings

    def save_embeddings(self, embeddings, relative_path):
        all_embeddings = np.vstack(embeddings)
        absolute_path = os.path.join(self.script_dir, relative_path)
        np.save(absolute_path, all_embeddings)

    def load_embeddings(self, relative_path):
        absolute_path = os.path.join(self.script_dir, relative_path)
        return np.load(absolute_path, allow_pickle=True)