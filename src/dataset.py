from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset

class TextActivationDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, model_name: str, max_length: int = 128,
                 layer_index: int = 6, num_samples: int = 1000):
        self.dataset = load_dataset(dataset_name, split=split)
        if num_samples:
            self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.max_length = max_length
        self.layer_index = layer_index
        self.activations = self._compute_activations()

    def _compute_activations(self):
        activations = []
        for item in self.dataset:
            inputs = self.tokenizer(
                item['text'],
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            with torch.no_grad():
                outputs = self.model(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )
                layer_activations = outputs.hidden_states[self.layer_index][0]
                mean_activation = layer_activations.mean(dim=0)
                activations.append(mean_activation)
        return torch.stack(activations)

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]