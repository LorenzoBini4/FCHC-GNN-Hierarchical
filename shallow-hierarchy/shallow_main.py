import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import random
import torch.nn as nn
import argparse
import sys
from model.shallow_model import *
from utils.fchc-train import *
from tqdm import tqdm

torch.cuda.empty_cache()
# Set seeds for reproducibility
seed_value = 77
random.seed(seed_value)
np.random.seed(seed_value)
# Set seed for PyTorch
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Set seed for CUDA operations (if available)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

PRINT_MEMORY = False
PRINT_STATEMENT = True

device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

torch.cuda.empty_cache()
if __name__ == "__main__": 
    print(sys.argv)
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("--graph_path", type=str, help="Path to the input graph file")
    parser.add_argument('--model', type=str, choices=['SAGE', 'GAT', 'GCN', 'DNN'], help='Choose GNN model: SAGE, GAT, or GCN')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--hidden_features', type=int, default=64, help='Number of hidden features')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--in_heads', type=int, default=8, help='Number of in heads (for GAT)')
    parser.add_argument('--out_heads', type=int, default=8, help='Number of out heads (for GAT)')
    parser.add_argument('--max_num_epochs', type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--start_lr', type=float, default=0.000001, help='Initial learning rate')
    parser.add_argument('--num_repetitions', type=int, default=4, help='Number of repetitions')
    args = parser.parse_args()

INPUT_PATH = 'data_flat_hierarchical'    
label0_count=[]
for j in range(19):  
    df = pd.read_csv(f"{INPUT_PATH}/Case_{j+1}.csv") 
    label0_count.append(len(df))
    
class MyGraphDataset(Dataset):
    def __init__(self, num_samples, graph_path, transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(transform, pre_transform)
        self.num_samples = num_samples
        self.data_list = torch.load(graph_path)
        #self.class_weights = class_weights_tensor

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data_list[idx]    

with ClearCache():
    def gnn_evaluation(gnn, max_num_epochs, batch_size, start_lr, num_repetitions, graph_path, min_lr=0.0000001, factor=0.05, patience=5, all_std=True):
        dataset = MyGraphDataset(num_samples=len(torch.load(graph_path)), graph_path=graph_path).shuffle()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        best_model_state_dict = None
        patient_dict=dict()
        for i in range(num_repetitions):
            kf = KFold(n_splits=7, shuffle=True)
            dataset.shuffle()

            for train_index, test_index in kf.split(list(range(len(dataset)))):
                train_index, val_index = train_test_split(train_index, test_size=0.1)

                train_dataset = dataset[train_index.tolist()]
                val_dataset = dataset[val_index.tolist()]
                test_dataset = dataset[test_index.tolist()]

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                num_patients = len(test_loader)
                print(f"Number of patients in the test loader: {num_patients}")

                input_dim = dataset[0].x.shape[1]
                output_dim = len(torch.unique(dataset[0].y))
                model = gnn(input_dim, output_dim, args).to(device)
                model.reset_parameters()

                optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=0.0007)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                                       patience=patience, min_lr=0.0000001)

                best_val_acc = 0.0
                best_model_state_dict = None
                early_stopping_counter = 0
                early_stopping_patience = 20

                for epoch in range(1, max_num_epochs + 1):
                    lr = scheduler.optimizer.param_groups[0]['lr']
                    torch.cuda.empty_cache()
                    train_shallow(train_loader, model, optimizer, device) #, class_weights_tensor)
                    val_acc = val_shallow(val_loader, model, device)
                    scheduler.step(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_val_acc = val(test_loader, model, device) * 100.0
                        best_model_state_dict = model.state_dict()
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= early_stopping_patience:
                            print("Early stopping triggered. No improvement in validation accuracy for {} epochs.".format(early_stopping_patience))
                            break

                model.load_state_dict(best_model_state_dict)
                torch.save(model.state_dict(), 'hc_bestflat_sage.pt')
                # Evaluate on the entire test set
                model.eval()
                for data in test_loader:
                    data = data.to(device)
                    output = model(data)
                    predss=output.max(dim=1)[1].cpu().numpy()
                    labelss=data.y.cpu().numpy()
                    idx=label0_count.index(len(labelss))+1
                    precision, recall, f1, _ = precision_recall_fscore_support(labelss, predss, average='weighted', zero_division=1)

                    if idx not in patient_dict.keys():
                        patient_dict[idx]=dict()
                        patient_dict[idx]['f1']=[f1]
                        patient_dict[idx]['pred']=[predss]
                        patient_dict[idx]['label']=[labelss]
                    else:
                        patient_dict[idx]['f1'].append(f1)
                        patient_dict[idx]['pred'].append(predss)
                        patient_dict[idx]['label'].append(labelss)

        return patient_dict

    def main(graph_path):
        patient_dict = gnn_evaluation(GNNModel, args.max_num_epochs, batch_size=1, start_lr=args.start_lr, num_repetitions=args.num_repetitions, graph_path=graph_path)
        average_ratio_per_label = []
        for key in patient_dict.keys():
                idx = patient_dict[key]['f1'].index(max(patient_dict[key]['f1']))
                df = pd.read_csv(f"{INPUT_PATH}/Case_{key}.csv")
                df['predicted label'] = patient_dict[key]['pred'][idx]
                if not os.path.exists("flat_prediction"):
                  os.makedirs("flat_prediction")
                df.to_csv(f"flat_prediction/Case_{key}.csv", index=False)

                conf_matrix = confusion_matrix(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])
                precision = precision_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
                recall = recall_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
                accuracy = accuracy_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])
                f1 = f1_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
                average_precision = np.mean([precision_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])
                average_recall = np.mean([recall_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])
                average_f1 = np.mean([f1_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys()])
                average_accuracy = np.mean([accuracy_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx]) for key in patient_dict.keys()])

                if PRINT_STATEMENT:
                    print(f"Metrics for Patient {key}:")
                    print(f"Confusion Matrix:\n{conf_matrix}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"F1 Score: {f1:.4f}")

                    print("\nAverage Metrics Across All Patients:")
                    print(f"Average Precision: {average_precision:.4f}")
                    print(f"Average Recall: {average_recall:.4f}")
                    print(f"Average F1 Score: {average_f1:.4f}")
                    print(f"Average Accuracy: {average_accuracy:.4f}")

                ratio_per_label = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
                correct_per_label = np.diag(conf_matrix)
                total_per_label = np.sum(conf_matrix, axis=1)

                for i, label in enumerate(range(conf_matrix.shape[0])):
                    ratio_correct = f"{correct_per_label[i]}/{total_per_label[i]}"
                    print(f"Label {label}:")
                    print(f"Percentage of Correct Predictions: {ratio_per_label[i]*100:.2f}%, corresponding to {ratio_correct}")
                    if len(average_ratio_per_label) <= i:
                        average_ratio_per_label.append([ratio_per_label[i]])
                    else:
                        average_ratio_per_label[i].append(ratio_per_label[i])
                print("-" * 50)
        average_ratio_per_label = np.mean(average_ratio_per_label, axis=1)
        # Print the average ratios
        print("\nAverage Ratios Across All Patients:")
        print(f"Average Accuracy: {average_accuracy:.4f}")
        for i, average_ratio in enumerate(average_ratio_per_label):
            print(f"Label {i}: {average_ratio:.4f}")

    main(args.graph_path)
