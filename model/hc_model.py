import torch
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, MessagePassing
import pandas as pd
import numpy as np 
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import argparse
import sys
import torch.nn as nn

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.empty_cache()

INPUT_PATH = 'data_hierarchical'
to_skip = ['root']
ATTRIBUTE_class = "1,1_1,1_1_1,1_1_1_1,1_1_1_2,1_1_2,1_1_3,1_1_3_1,1_1_3_2,1_2,1_3,2"
input_dim = 12
output_dim= len(set(ATTRIBUTE_class.split(',')))+1

g = nx.DiGraph()
for branch in ATTRIBUTE_class.split(','):
    term = branch.split('_')
    if len(term)==1:
        g.add_edge(term[0], 'root')
    else:
        for i in range(2, len(term) + 1):
            g.add_edge('.'.join(term[:i]), '.'.join(term[:i-1]))
nodes = sorted(g.nodes(), key=lambda x: (len(x.split('.')),x))
nodes_idx = dict(zip(nodes, range(len(nodes))))
g_t = g.reverse()
evall = [t not in to_skip for t in nodes]
AA = np.array(nx.to_numpy_array(g, nodelist=nodes))
R = np.zeros(AA.shape)
np.fill_diagonal(R, 1)
gg = nx.DiGraph(AA) # train.A is the matrix where the direct connections are stored 
for i in range(len(AA)):
    ancestors = list(nx.descendants(gg, i)) # Here we need to use the function nx.descendants() because in the directed 
            # graph the edges have source from the descendant and point towards the ancestor 
    if ancestors:
        R[i, ancestors] = 1
R = torch.tensor(R)
#Transpose to get the descendants for each node 
R = R.transpose(1, 0)

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.empty_cache()
    
total_count=[]
for j in range(19):  
    df = pd.read_csv(f"{INPUT_PATH}/Case_{j+1}.csv", low_memory=False)
    total_count.append(len(df))

def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out

############################################## FCHCGNN PLUG-IN MODULES ########################################################
# Define GATLayer
class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, concat_param, dropout_param):
        super(GATLayer, self).__init__()
        self.conv = GATConv(input_dim, output_dim, heads=num_heads, concat=concat_param, dropout=dropout_param)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

# Define HCGAT
class HCFCGAT(nn.Module):
    def __init__(self, R, input_dim, output_dim, hidden_dim, num_heads, out_heads, num_layers, dropout):
        super(HCFCGAT, self).__init__()
        self.R = R
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_heads = out_heads
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        gat_layers = []
        for i in range(self.num_layers):
            if i == 0:
                gat_layers.append(GATLayer(self.input_dim, self.hidden_dim, self.num_heads, True, dropout))
            elif i == self.num_layers - 1:
                gat_layers.append(GATLayer(self.hidden_dim * self.num_heads, self.output_dim, self.out_heads, False, dropout))
            else:
                gat_layers.append(GATLayer(self.hidden_dim * self.num_heads, self.hidden_dim, self.num_heads, True, dropout))
        self.gat_layers = nn.ModuleList(gat_layers)
       
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.f = nn.ReLU()
        self.reset_parameters()  # Initialize the weights

    def reset_parameters(self):
        for gat_layer in self.gat_layers:
            gat_layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.f(x)
                x = self.drop(x)
            else:
                x = self.sigmoid(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

# Define HCSAGE
class HCFCSAGE(nn.Module):
    def __init__(self, R, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(HCFCSAGE, self).__init__()
        self.R = R
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        sage_layers = []
        for i in range(self.num_layers):
            if i == 0:
                sage_layers.append(SAGEConv(self.input_dim, self.hidden_dim))
            elif i == self.num_layers - 1:
                sage_layers.append(SAGEConv(self.hidden_dim, self.output_dim))
            else:
                sage_layers.append(SAGEConv(self.hidden_dim, self.hidden_dim))
        self.sage_layers = nn.ModuleList(sage_layers)

        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.f = nn.ReLU()
        self.reset_parameters()  # Initialize the weights

    def reset_parameters(self):
        for sage_layer in self.sage_layers:
            sage_layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.sage_layers[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.f(x)
                x = self.drop(x)
            else:
                x = self.sigmoid(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

# Define HCDNN
class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return F.relu(self.fc(x))

class HCFCDNN(nn.Module):
    def __init__(self, R, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(HCFCDNN, self).__init__()
        self.R = R
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        dnn_layers = []
        for i in range(self.num_layers):
            if i == 0:
                dnn_layers.append(FullyConnectedLayer(self.input_dim, self.hidden_dim))
            elif i == self.num_layers - 1:
                dnn_layers.append(FullyConnectedLayer(self.hidden_dim, self.output_dim))
            else:
                dnn_layers.append(FullyConnectedLayer(self.hidden_dim, self.hidden_dim))
        self.dnn_layers = nn.ModuleList(dnn_layers)

        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.f = nn.ReLU()
        self.reset_parameters()  

    def reset_parameters(self):
        for dnn_layer in self.dnn_layers:
            dnn_layer.reset_parameters()

    def forward(self, data):
        x = data.x
        for i in range(self.num_layers):
            x = self.dnn_layers[i](x)
            if i != self.num_layers - 1:
                x = self.f(x)
                x = self.drop(x)
        x = self.sigmoid(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

# Define HCGNN
class GNNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(GNNLayer, self).__init__(aggr='add')
        self.fc = nn.Linear(input_dim, output_dim)
        self.reset_parameters()  # Initialize the weights

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.fc(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return x_j * norm.view(-1, 1)

class HCFCGNN(nn.Module):
    def __init__(self, R, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(HCFCGNN, self).__init__()
        self.R = R
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        gnn_layers = []
        for i in range(self.num_layers):
            if i == 0:
                gnn_layers.append(GNNLayer(self.input_dim, self.hidden_dim))
            elif i == self.num_layers - 1:
                gnn_layers.append(GNNLayer(self.hidden_dim, self.output_dim))
            else:
                gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim))
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.f = nn.ReLU()
        self.reset_parameters()  # Initialize the weights

    def reset_parameters(self):
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.gnn_layers[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.f(x)
                x = self.drop(x)
            else:
                x = self.sigmoid(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out

# Define HCGCN
class HCFCGCN(nn.Module):
    def __init__(self, R, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(HCFCGCN, self).__init__()
        self.R = R
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        gcn_layers = []
        for i in range(self.num_layers):
            if i == 0:
                gcn_layers.append(GCNConv(self.input_dim, self.hidden_dim))
            elif i == self.num_layers - 1:
                gcn_layers.append(GCNConv(self.hidden_dim, self.output_dim))
            else:
                gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.f = nn.ReLU()
        self.reset_parameters()  # Initialize the weights

    def reset_parameters(self):
        for gcn_layer in self.gcn_layers:
            gcn_layer.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.gcn_layers[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.f(x)
                x = self.drop(x)
            else:
                x = self.sigmoid(x)
        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out
############################################## FCHCGNN PLUG-IN MODULES ########################################################