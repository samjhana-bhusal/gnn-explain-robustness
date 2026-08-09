"""
GNN backbones for the benchmark.

Five architectures behind one constructor so the whole pipeline is backbone-
agnostic: GCN, GAT, GraphSAGE, APPNP, and SGC. SGC is linear in the propagated
features, which makes it the closed-form sanity check for the certificate's
theory (the smoothed inclusion probabilities have an analytic form there).

Widened from the prototype's hidden=16 to hidden=64 for the non-linear models;
kept a two-layer / K=2 receptive field so the locality lemma uses K=2 by default.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, APPNP, SGConv

BACKBONES = ("GCN", "GAT", "GraphSAGE", "APPNP", "SGC")


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        model_type: str = "GCN",
        dropout: float = 0.5,
        heads: int = 4,
        K: int = 2,
    ):
        super().__init__()
        self.model_type = model_type
        self.dropout = dropout
        self.K = K  # receptive-field depth used by surface.py

        if model_type == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == "GAT":
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        elif model_type == "GraphSAGE":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif model_type == "APPNP":
            # Linear encoder + personalised PageRank propagation (K steps).
            self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
            self.prop = APPNP(K=K, alpha=0.1)
        elif model_type == "SGC":
            # Single linear layer over K-step propagated features.
            self.conv1 = SGConv(in_channels, out_channels, K=K, cached=False)
        else:
            raise ValueError(f"Unknown model_type {model_type!r}. Known: {BACKBONES}")

    def forward(self, x, edge_index, edge_weight=None, *args, **kwargs):
        # edge_weight is threaded through where the conv supports it (GCN/SGC/
        # APPNP); the gradient explainer uses it as a differentiable per-edge gate.
        if self.model_type == "APPNP":
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            x = self.prop(x, edge_index, edge_weight=edge_weight)
            return F.log_softmax(x, dim=1)
        if self.model_type == "SGC":
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            return F.log_softmax(x, dim=1)
        if self.model_type == "GCN":
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            return F.log_softmax(x, dim=1)

        # GAT / GraphSAGE: no scalar edge_weight support in the standard convs.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


@torch.no_grad()
def accuracy(model, data, mask) -> float:
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred[mask] == data.y[mask]
    return int(correct.sum()) / int(mask.sum())


def train_model(
    model, data, *, lr: float = 0.01, weight_decay: float = 5e-4,
    epochs: int = 200, patience: int = 50, verbose: bool = False,
):
    """Train with early stopping on val accuracy. Returns best test accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_test, best_state, wait = 0.0, 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val = accuracy(model, data, data.val_mask)
        test = accuracy(model, data, data.test_mask)
        if val > best_val:
            best_val, best_test, wait = val, test, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1:03d} | loss {loss.item():.4f} | val {val:.4f} | test {test:.4f}")
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_test
