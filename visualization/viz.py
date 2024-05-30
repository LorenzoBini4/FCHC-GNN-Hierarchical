######### EMBEDDINGS WITH t-SNE #########
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score
import torch
from torch_geometric.explain import Explainer
from torch_geometric.explain import GNNExplainer, AttentionExplainer
from hc_model import *
import seaborn as sns

model = HCGAT(R)
model.load_state_dict(torch.load('best_pt/best_gat.pt', map_location='cpu')) # if it has been saved on GPU
model.eval()

masked_graphs = torch.load(INPUT_GRAPH)
data7 = masked_graphs[6]  # Patient7, since PyG list starts from zero
out = model(data7.x, data7.edge_index)
#z = torch.load('z_embeddings.pt')

# ################################### tSNE VISUALIZATION ###################################
def visualize(h, color, legend_labels, cmap_name="Set2"):
    z = TSNE(n_components=2, perplexity = 10).fit_transform(h.detach().cpu().numpy())
    #torch.save(z, 'z_embeddings.pt')
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(z[:, 0], z[:, 1], s=0.3, c=color, cmap=cmap_name)

    # Create a legend with corresponding colors
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10, label=label) for i, label in enumerate(legend_labels)]
    plt.legend(handles=handles)

    plt.savefig('tsne_embeddings_with_legend.png', format='png', dpi=400)
    plt.show()

a = set(data7.yy)
a = {x:i for i,x in enumerate(a)}
color = [a[x] for x in data7.yy]
legend_labels = ['NK Cells', 'CD4 T Cells', 'Kappa Pos', 'Lambda Pos', 'Monocytes', 'CD8 T Cells', 'Neutrophils']

#visualize(z, color=color, legend_labels=legend_labels)
visualize(out, color=color, legend_labels=legend_labels)

################################## Seaborn tSNE VISUALIZATION ###################################
#Set up Seaborn style
sns.set(style="whitegrid")
def visualize(h, color, legend_labels, cmap_name="tab10"):
    z = TSNE(n_components=2, perplexity=10).fit_transform(h.detach().cpu().numpy())
    #torch.save(z, 'z_embeddings.pt')
    plt.figure(figsize=(12, 10))
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_facecolor('white')  # Set the background color to black

    # Create the scatter plot
    scatter = plt.scatter(z[:, 0], z[:, 1], s=3, c=color, cmap=cmap_name, alpha=0.9, edgecolors='w', linewidth=0.1)

    plt.title('t-SNE Visualization of Cell Types', fontsize=20, color='white')
    plt.xlabel('t-SNE Component 1', fontsize=14, color='white')
    plt.ylabel('t-SNE Component 2', fontsize=14, color='white')

    # Create a legend with corresponding colors
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10, label=label) for i, label in enumerate(legend_labels)]
    plt.legend(handles=handles, title='Cell Types', fontsize=12, title_fontsize=14, loc='upper right', frameon=True, shadow=True, borderpad=1)

    sns.despine(left=True, bottom=True)
    plt.savefig('tsne_seaborn_embeddings_with_legend.png', format='png', dpi=400, bbox_inches='tight')
    plt.show()

# Map labels to color indices
a = set(data7.yy)
a = {x: i for i, x in enumerate(a)}
color = [a[x] for x in data7.yy]
legend_labels = ['Neutrophils', 'CD8 T Cells', 'Lambda Pos', 'NK Cells', 'Kappa Pos', 'Monocytes', 'CD4 T Cells'] 

#visualize(z, color=color, legend_labels=legend_labels)
visualize(out, color=color, legend_labels=legend_labels)
#exit(0)

################################### EXPLAINABILITY ###################################
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

#data = masked_graphs[7]
node_index = 7142 # Select the node index on your choice
explanation = explainer(data7.x, data7.edge_index, index=node_index) # modify the forward function to accept x and edge_index

print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance_7.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph_7.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")

######################################## ATTENTION EXPLAINER ########################################
# AttentionExplainer uses attention coefficients to determine edge weights/opacities.
attention_explainer = Explainer(
    model=model,
    # AttentionExplainer takes an optional reduce parameter. The reduce parameter
    # allows you to set how you want to aggregate attention coefficients over layers
    # and heads. The explainer will then aggregate these values using this
    # given method to determine the edge_mask (we use the default 'max' here).
    algorithm=AttentionExplainer(),
    explanation_type='model',
    # Like PGExplainer, AttentionExplainer also does not support node_mask_type
    edge_mask_type='object',
    model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs'),
)

node_index=10

attention_explanation = attention_explainer(data7.x, data7.edge_index, index=node_index)
attention_explanation.visualize_graph("attention_graph_7.png", backend="graphviz")
plt.imshow(plt.imread("attention_graph_7.png"))

################################### DEGREES VISUALIZATION ##########################
from torch_geometric.utils import degree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Get model's classifications
out = out.cpu()
data7.y = data7.y.cpu().numpy() # Convert data7.y to a numpy array
degrees = degree(data7.edge_index[0]).cpu().numpy()
accuracies = []
sizes = []
# Convert out to numpy array
out_labels = torch.nn.functional.one_hot(out.argmax(dim=1)).numpy()

# Accuracy for degrees between 0 and 5
for i in range(0, 6):
    mask = np.where(degrees == i)[0]
    if len(mask) > 0:
        accuracies.append(accuracy_score(out_labels[mask], data7.y[mask]))
    else:
        accuracies.append(0)
    sizes.append(len(mask))

# Accuracy for degrees > 5
mask = np.where(degrees > 5)[0]
if len(mask) > 0:
    accuracies.append(accuracy_score(out_labels[mask], data7.y[mask]))
else:
    accuracies.append(0)
sizes.append(len(mask))

fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlabel('Node degree')
ax.set_ylabel('Accuracy score')
ax.set_facecolor('#EFEEEA')
plt.bar(['0', '1', '2', '3', '4', '5', '>5'],
        accuracies,
        color='#0A047A')
for i in range(0, 7):
    plt.text(i, accuracies[i], f'{accuracies[i]*100:.2f}%',
             ha='center', color='#0A047A')
for i in range(0, 7):
    plt.text(i, accuracies[i] / 2, sizes[i],
             ha='center', color='white')

plt.savefig('degree_accuracy.png', dpi=100)
#########################
