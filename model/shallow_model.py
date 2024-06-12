import torch
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GATConv,SAGEConv, GCNConv
import torch.nn as nn
import argparse

class FCHCGNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(FCHCGNN, self).__init__()
        self.num_layers = args.num_layers
        self.hidden_features = args.hidden_features
        self.dropout = args.dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.args = args
        if args.model == 'FCHCSAGE':
            self.conv = self.make_sage_layers(input_dim, output_dim)
        elif args.model == 'FCHCGAT':
            self.conv = self.make_gat_layers(input_dim, output_dim)
        elif args.model == 'FCHCGCN':
            self.conv = self.make_gcn_layers(input_dim, output_dim)
        elif args.model == 'FCHCDNN':
            self.conv = self.make_dnn_layers(input_dim, output_dim)
   
    def make_sage_layers(self, input_dim, output_dim):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(SAGEConv(input_dim, self.hidden_features))
            else:
                layers.append(SAGEConv(self.hidden_features, self.hidden_features))
        layers.append(SAGEConv(self.hidden_features, output_dim))
        return torch.nn.ModuleList(layers)
    
    def make_gat_layers(self, input_dim, output_dim):
        args = self.args
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(GATConv(input_dim, self.hidden_features, heads=args.in_heads, dropout=self.dropout))
            else:
                layers.append(GATConv(self.hidden_features * args.in_heads, self.hidden_features, heads=args.out_heads, dropout=self.dropout))
        layers.append(GATConv(self.hidden_features * args.in_heads, output_dim))
        return torch.nn.ModuleList(layers)
    
    def make_gcn_layers(self, input_dim, output_dim):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(GCNConv(input_dim, self.hidden_features))
            else:
                layers.append(GCNConv(self.hidden_features, self.hidden_features))
        layers.append(GCNConv(self.hidden_features, output_dim))
        return torch.nn.ModuleList(layers)
    
    def make_dnn_layers(self, input_dim, output_dim):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, self.hidden_features))
            else:
                layers.append(nn.Linear(self.hidden_features, self.hidden_features))
        layers.append(nn.Linear(self.hidden_features, output_dim))
        return torch.nn.ModuleList(layers)
    
    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv:
            if isinstance(conv, nn.Linear):
                x = F.relu(conv(x))
            else:
                x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")
    args = parser.parse_args()
