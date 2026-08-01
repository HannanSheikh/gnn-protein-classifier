"""
Train a GIN model on the PROTEINS graph classification dataset.

Usage:
    python train.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from model import GIN

EPOCHS = 100
HIDDEN_CHANNELS = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.01
TRAIN_FRACTION = 0.8


def load_data():
    dataset = TUDataset(root="data/TUDataset", name="PROTEINS")
    dataset = dataset.shuffle()

    train_size = int(len(dataset) * TRAIN_FRACTION)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, test_loader, dataset.num_features, dataset.num_classes, len(dataset)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    for batch in loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
    return correct / len(loader.dataset)


def main():
    train_loader, test_loader, num_features, num_classes, total_graphs = load_data()
    print(f"Loaded PROTEINS: {total_graphs} protein graphs, {num_features} node features, {num_classes} classes")

    model = GIN(in_channels=num_features, hidden_channels=HIDDEN_CHANNELS, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer)
        if epoch % 10 == 0 or epoch == 1:
            train_acc = evaluate(model, train_loader)
            test_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch:03d} | loss {loss:.4f} | train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    print(f"\nFinal test accuracy: {evaluate(model, test_loader):.3f}")


if __name__ == "__main__":
    main()
