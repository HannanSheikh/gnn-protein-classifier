# GNN Graph Classification on Protein Structure Graphs (PROTEINS)

Small project applying a graph neural network to a biomedical graph
classification task, using the PROTEINS dataset.

## What this is

PROTEINS is a dataset where each protein is represented as a graph nodes are
secondary structure elements (helix, sheet, turn), edges connect elements that
are close together in the protein's 3D structure. The task is to predict
whether each protein is an enzyme or not, based on this graph structure.

I built this to get hands-on practice with graph neural networks applied to
biological data, since a lot of biomedical data (protein interactions, cell
networks, molecular structures) is naturally graph-shaped rather than
grid-shaped like images.

## Model

Uses a GIN (Graph Isomorphism Network) with global pooling. GIN is a common
choice for graph classification specifically (as opposed to node
classification) because it tends to be better at telling different graph
structures apart.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

```bash
python src/train.py
```

Downloads PROTEINS automatically on first run, trains the model, prints
train/test accuracy every 10 epochs.

## Structure

```
gnn-protein-classifier/
├── README.md
├── requirements.txt
└── src/
    ├── model.py     # GIN model
    └── train.py     # training/eval loop
```
