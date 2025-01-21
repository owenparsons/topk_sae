import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_sparse_autoencoder(model, dataloader, num_epochs, learning_rate=1e-3,
                              aux_loss_weight=0.1, dead_feature_threshold=1000, grad_clip=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    stats = {'reconstruction_loss': [], 'aux_loss': [], 'total_loss': []}

    for epoch in range(num_epochs):
        epoch_stats = {'reconstruction_loss': 0., 'aux_loss': 0., 'total_loss': 0.}
        num_batches = 0
        for batch in dataloader:
            x = batch.to(model.device) if isinstance(batch, torch.Tensor) else batch[0].to(model.device)
            sparse_code, top_k_values, top_k_indices = model.encode(x)
            reconstruction = model.decode(sparse_code)
            recon_loss = F.mse_loss(reconstruction, x)

            model.usage_count *= 0.99
            model.usage_count.scatter_add_(0, top_k_indices.flatten(),
                                           torch.ones_like(top_k_indices.flatten(), dtype=torch.float))

            dead_features = model.usage_count < (1 / dead_feature_threshold)
            aux_loss = -aux_loss_weight * sparse_code[:, dead_features].abs().mean() if dead_features.any() else torch.tensor(0., device=model.device)
            total_loss = recon_loss + aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                norm = model.decoder.weight.norm(dim=1, keepdim=True)
                model.decoder.weight.data /= norm

            epoch_stats['reconstruction_loss'] += recon_loss.item()
            epoch_stats['aux_loss'] += aux_loss.item()
            epoch_stats['total_loss'] += total_loss.item()
            num_batches += 1

        for key in epoch_stats:
            epoch_stats[key] /= num_batches
            stats[key].append(epoch_stats[key])

        print(f"Epoch {epoch+1}/{num_epochs} - Recon Loss: {epoch_stats['reconstruction_loss']:.6f}, "
              f"Aux Loss: {epoch_stats['aux_loss']:.6f}, Total Loss: {epoch_stats['total_loss']:.6f}")

    return stats