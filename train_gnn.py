import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric.transforms as T

# Simple GNN model supporting GCN or GAT
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='GCN'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
            # Input dimension to conv2 is hidden_channels * 4 because of concat=True with 4 heads
            self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)
            
    def forward(self, x, edge_index, *args, **kwargs):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, data, optimizer, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            acc = test(model, data)
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f}")

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc

def main():
    print("=== Training GNN Model ===")
    
    # Store dataset inside workspace under 'data' directory
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("Loading Cora dataset...")
    # T.NormalizeFeatures is standard for Cora
    dataset = Planetoid(root=data_dir, name="Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    
    print(f"Dataset: {dataset.name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Train GCN
    gcn_model = GNN(
        in_channels=dataset.num_features,
        hidden_channels=16,
        out_channels=dataset.num_classes,
        model_type='GCN'
    )
    
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)
    print("\nTraining GCN model...")
    train(gcn_model, data, optimizer, epochs=150)
    gcn_acc = test(gcn_model, data)
    print(f"GCN Final Test Accuracy: {gcn_acc:.4f}")
    
    # Save model checkpoint
    model_path = os.path.join(data_dir, "gcn_cora.pt")
    torch.save(gcn_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
if __name__ == "__main__":
    main()
