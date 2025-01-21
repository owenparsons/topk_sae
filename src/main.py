import torch
from torch.utils.data import DataLoader
from dataset import TextActivationDataset
from model import TopKSparseAutoencoder
from training import train_sparse_autoencoder

if __name__ == "__main__":
    dataset_name = "ag_news"
    model_name = "distilbert-base-uncased"
    num_samples = 1000
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    device="cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    dataset = TextActivationDataset(
        dataset_name=dataset_name,
        split='train',
        model_name=model_name,
        num_samples=num_samples,
        layer_index=3
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.activations.shape[1]
    print("Initializing model...")
    model = TopKSparseAutoencoder(
        input_dim=input_dim,
        dict_size=input_dim * 4,
        k=32,
        device=device
    )

    print("Starting training...")
    stats = train_sparse_autoencoder(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )