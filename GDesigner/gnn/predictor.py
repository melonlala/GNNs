import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Predictor(nn.Module):

    def __init__(self, input_dim, hidden_dim,n_layers=2,dropout=0.5):
        super(Predictor, self).__init__()
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
        for i in range(n_layers-1):
            self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
        self.dropout = dropout
        self.projector = nn.Linear(input_dim, hidden_dim)

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()
        self.projector.reset_parameters()
    
    def forward(self, node_embedding, task_embedding, edge_index):
        for i, gnn in enumerate(self.gnn_layers):
            node_embedding = gnn(node_embedding, edge_index)
            if i!= len(self.gnn_layers)-1:
                node_embedding = F.relu(x)
                node_embedding = F.dropout(x, p=self.dropout, training=self.training)
        node_embedding = F.normalize(node_embedding, p=2, dim=1)
        node_embedding = node_embedding.mean(dim=0)
        task_embedding = self.projector(task_embedding)
        score = torch.sigmoid(torch.sum(node_embedding*task_embedding, dim=1))
        return score
        