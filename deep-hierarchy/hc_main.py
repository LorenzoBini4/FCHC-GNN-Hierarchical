import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import random
import os
from tqdm import tqdm
from utils.fchc-train import *
from model.hc_model import *
import argparse
import time

# Set seeds for reproducibility
seed_value = 77
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
R = R.unsqueeze(0).to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for hierarchical HCFCGNN models")
    parser.add_argument("--graph_path", type=str, help="Path to the input graph file")
    parser.add_argument('--model', type=str, required=True, choices=['FCHCGAT', 'FCHCSAGE', 'FCHCDNN', 'FCHCGNN', 'FCHCGCN'], help='Model to use')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--hidden_features', type=int, default=64, help='Number of hidden features')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--start_lr', type=float, default=0.001, help='Starting learning rate')
    parser.add_argument('--max_num_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--num_repetitions', type=int, default=4, help='Number of repetitions')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads (only for GAT)')
    parser.add_argument('--out_heads', type=int, default=2, help='Number of output heads (only for GAT)')
    args = parser.parse_args()
    print(sys.argv)

class MyGraphDataset(Dataset):
    def __init__(self,  num_samples, graph_path, transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(transform, pre_transform)
        self.num_samples = num_samples
        self.data_list = torch.load(graph_path)   

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data_list[idx]

total_count=[]
for j in range(19):  
    df = pd.read_csv(f"{INPUT_PATH}/Case_{j+1}.csv", low_memory=False)
    total_count.append(len(df))
        
with ClearCache():
    def gnn_evaluation(gnn, max_num_epochs, batch_size, start_lr, num_repetitions, graph_path, min_lr=0.000001, factor=0.05, patience=7, all_std=True):
        dataset = MyGraphDataset(num_samples=len(torch.load(graph_path)), graph_path=graph_path).shuffle()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        patient_dict = dict()
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
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                num_patients = len(test_loader)
                print(f"Number of patients in the test loader: {num_patients}")

                if gnn.__name__ == 'FCHCGAT':
                    model = gnn(R, input_dim, output_dim, hidden_dim=args.hidden_features,
                                num_heads=args.num_heads, out_heads=args.out_heads, num_layers=args.num_layers,
                                dropout=args.dropout).to(device)
                else:
                    model = gnn(R, input_dim, output_dim, hidden_dim=args.hidden_features,
                                num_layers=args.num_layers, dropout=args.dropout).to(device)
                    
                model.reset_parameters()
                optimizer = torch.optim.Adam(model.parameters(), lr=start_lr) #weight_decay=0.0001
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                                    patience=patience, min_lr=0.0000001)
                criterion = nn.BCELoss()
                best_val_acc = 0.0
                early_stopping_counter = 0
                early_stopping_patience = 7

                best_model = None
                best_val_acc = 0.0
                for epoch in range(1, max_num_epochs + 1):
                    lr = scheduler.optimizer.param_groups[0]['lr']
                    torch.cuda.empty_cache()
                    train_deep(train_loader, model, optimizer, device, criterion)
                    torch.cuda.empty_cache()
                    val_acc = val_deep(val_loader, model, device)
                    scheduler.step(val_acc)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = model.state_dict()
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= early_stopping_patience:
                            print("Early stopping triggered. No improvement in validation accuracy for {} epochs.".format(
                                early_stopping_patience))
                            break

                # Generate timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")

                # Save best model with a unique name
                model_name = f"best_FCHCGNN_{timestamp}.pt"
                torch.save(best_model, model_name)
                
                # Evaluate on the entire test set
                with torch.no_grad():
                    model.eval()
                    torch.cuda.empty_cache()
                    for data in tqdm(test_loader, desc='Testing'):
                        model.eval()
                        data = data.to(device)
                        constrained_output = model(data)
                        predss = constrained_output.data
                        cell_preds = []
                        for row in predss:
                            ind_max = row[:2].argmax().item() + 1
                            if ind_max == 1:
                                ind_n = row[3:6].argmax().item() + 1
                                if ind_n == 1:
                                    ind_nn = row[6:9].argmax().item() + 1
                                    if ind_nn == 1:
                                        ind_nnn = row[9:11].argmax().item() + 1
                                        cell_preds.append(str(ind_max) + '_' + str(ind_n) + '_' + str(ind_nn) + '_' + str(
                                            ind_nnn))
                                    elif ind_nn == 3:
                                        ind_nnn = row[11:].argmax().item() + 1
                                        cell_preds.append(str(ind_max) + '_' + str(ind_n) + '_' + str(ind_nn) + '_' + str(
                                            ind_nnn))
                                    else:
                                        cell_preds.append(str(ind_max) + '_' + str(ind_n) + '_' + str(ind_nn))
                                else:
                                    cell_preds.append(str(ind_max) + '_' + str(ind_n))
                            else:
                                cell_preds.append(str(ind_max))
                        predsss = cell_preds
                        labelss = data.yy[0]
                        idx = total_count.index(len(labelss)) + 1

                        # Compute confusion matrix, precision, recall, and F1 score
                        alpha = [0] * len(predsss)
                        beta = [0] * len(predsss)
                        intersect = 0
                        tot_alpha = 0
                        tot_beta = 0
                        pred_matrix = np.zeros((len(predsss), len(nodes) - 1))
                        true_matrix = np.zeros((len(labelss), len(nodes) - 1))
                        for ii in range(len(predsss)):
                            if predsss[ii] == '2':
                                pred_matrix[ii, :] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            elif predsss[ii] == '1_2':
                                pred_matrix[ii, :] = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                            elif predsss[ii] == '1_3':
                                pred_matrix[ii, :] = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                            elif predsss[ii] == '1_1_2':
                                pred_matrix[ii, :] = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                            elif predsss[ii] == '1_1_1_1':
                                pred_matrix[ii, :] = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
                            elif predsss[ii] == '1_1_1_2':
                                pred_matrix[ii, :] = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
                            elif predsss[ii] == '1_1_3_1':
                                pred_matrix[ii, :] = [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                            elif predsss[ii] == '1_1_3_2':
                                pred_matrix[ii, :] = [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
                        for ii in range(len(labelss)):
                            if labelss[ii] == '2':
                                true_matrix[ii, :] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            elif labelss[ii] == '1_2':
                                true_matrix[ii, :] = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                            elif labelss[ii] == '1_3':
                                true_matrix[ii, :] = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                            elif labelss[ii] == '1_1_2':
                                true_matrix[ii, :] = [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                            elif labelss[ii] == '1_1_1_1':
                                true_matrix[ii, :] = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
                            elif labelss[ii] == '1_1_1_2':
                                true_matrix[ii, :] = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
                            elif labelss[ii] == '1_1_3_1':
                                true_matrix[ii, :] = [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                            elif labelss[ii] == '1_1_3_2':
                                true_matrix[ii, :] = [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
                        for ii in range(len(predsss)):
                            # define alpha
                            if predsss[ii] == '2':
                                alpha[ii] = ['root', predsss[ii]]
                            elif predsss[ii] == '1_2':
                                alpha[ii] = ['root', predsss[ii], '1']
                            elif predsss[ii] == '1_3':
                                alpha[ii] = ['root', predsss[ii], '1']
                            elif predsss[ii] == '1_1_2':
                                alpha[ii] = ['root', predsss[ii], '1', '1_1']
                            elif predsss[ii] == '1_1_1_1':
                                alpha[ii] = ['root', predsss[ii], '1', '1_1', '1_1_1']
                            elif predsss[ii] == '1_1_1_2':
                                alpha[ii] = ['root', predsss[ii], '1', '1_1', '1_1_1']
                            elif predsss[ii] == '1_1_3_1':
                                alpha[ii] = ['root', predsss[ii], '1', '1_1', '1_1_3']
                            elif predsss[ii] == '1_1_3_2':
                                alpha[ii] = ['root', predsss[ii], '1', '1_1', '1_1_3']
                            # define beta
                            if labelss[ii] == '2':
                                beta[ii] = ['root', labelss[ii]]
                            elif labelss[ii] == '1_2':
                                beta[ii] = ['root', labelss[ii], '1']
                            elif labelss[ii] == '1_3':
                                beta[ii] = ['root', labelss[ii], '1']
                            elif labelss[ii] == '1_1_2':
                                beta[ii] = ['root', labelss[ii], '1', '1_1']
                            elif labelss[ii] == '1_1_1_1':
                                beta[ii] = ['root', labelss[ii], '1', '1_1', '1_1_1']
                            elif labelss[ii] == '1_1_1_2':
                                beta[ii] = ['root', labelss[ii], '1', '1_1', '1_1_1']
                            elif labelss[ii] == '1_1_3_1':
                                beta[ii] = ['root', labelss[ii], '1', '1_1', '1_1_3']
                            elif labelss[ii] == '1_1_3_2':
                                beta[ii] = ['root', labelss[ii], '1', '1_1', '1_1_3']
                            intersect += len(list(set(alpha[ii]) & set(beta[ii])))
                            tot_alpha += len(alpha[ii])
                            tot_beta += len(beta[ii])
                        hP = intersect / tot_alpha
                        hR = intersect / tot_beta
                        hF = 2 * hP * hR / (hP + hR)
                        if idx not in patient_dict.keys():
                            patient_dict[idx] = dict()
                            patient_dict[idx]['precision'] = [hP]
                            patient_dict[idx]['recall'] = [hR]
                            patient_dict[idx]['F-score'] = [hF]
                            patient_dict[idx]['pred'] = [predsss]
                            patient_dict[idx]['label'] = [labelss]
                            patient_dict[idx]['pred_matrix'] = [pred_matrix]
                            patient_dict[idx]['true_matrix'] = [true_matrix]
                            # patient_dict[idx]['K_label']=[true_K_label / total_K_label]
                        else:
                            patient_dict[idx]['precision'].append(hP)
                            patient_dict[idx]['recall'].append(hR)
                            patient_dict[idx]['F-score'].append(hF)
                            patient_dict[idx]['pred'].append(predsss)
                            patient_dict[idx]['label'].append(labelss)
                            patient_dict[idx]['pred_matrix'].append(pred_matrix)
                            patient_dict[idx]['true_matrix'].append(true_matrix)
                            # patient_dict[idx]['K_label'].append(true_K_label / total_K_label)
        return patient_dict

    def main(graph_path, model_name, args):
        if model_name == 'FCHCGAT':
            model = FCHCGAT
        elif model_name == 'FCHCSAGE':
            model = FCHCSAGE
        elif model_name == 'FCHCDNN':
            model = FCHCDNN
        elif model_name == 'FCHCGNN':
            model = FCHCGNN
        elif model_name == 'FCHCGCN':
            model = FCHCGCN
        else:
            raise ValueError("Invalid model name")

        # Call gnn_evaluation with the selected model
        patient_dict = gnn_evaluation(model, args.max_num_epochs, batch_size=1, start_lr=args.start_lr,
                                    num_repetitions=args.num_repetitions, graph_path=graph_path, all_std=True)

        # Initialize a list to store the ratios for each label across all patients
        average_ratio_per_label = []
        F_scores = np.zeros((19, args.num_repetitions))
        no = 0
        for key in patient_dict.keys():
            F_scores[no,:] = patient_dict[key]['F-score']
            no += 1
        F_scores_mean = F_scores.mean(axis = 0)
        idx = F_scores_mean.argmax(axis=0)
        nodes.remove('root')

        for key in patient_dict.keys():
            df=pd.read_csv(f"{INPUT_PATH}/Case_{key}.csv", low_memory=False)  # Set low_memory=False to fix the warning
            df['predicted label']=patient_dict[key]['pred'][idx]
            #if not os.path.exists("new_HC_predicted"):
            #    os.makedirs("new_HC_predicted")
            #df.to_csv(f"new_HC_predicted/Case_{key}.csv", index=False)
            precision = patient_dict[key]['precision'][idx]
            recall = patient_dict[key]['recall'][idx]
            f1 = patient_dict[key]['F-score'][idx]
            print(f"Metrics for Patient {key}:")   
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            conf_mat_dict={}
            for label_col in range(len(nodes)):
                y_true_label = patient_dict[key]['true_matrix'][idx][:, label_col]
                y_pred_label = patient_dict[key]['pred_matrix'][idx][:, label_col]
                conf_mat_dict[nodes[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
                precision, recall, f1, _ = precision_recall_fscore_support(y_true_label, y_pred_label, average='weighted', zero_division=1)
                print(f"Label {label_col}:")
                print(f"Ratio of Correct Predictions: {recall:.4f}")
                if len(average_ratio_per_label) <= label_col:
                    average_ratio_per_label.append([recall])
                else:
                    average_ratio_per_label[label_col].append(recall)
            print("-" * 50)

        average_ratio_per_label = np.mean(average_ratio_per_label, axis=1)
        # Print the average ratios 
        print("\nAverage Ratios Across All Patients:")
        label_dict = {0: 'B', 1: 'M', 2: 'C', 3: 'E', 4: 'D', 5: 'F', 6: 'J',7:'G',8:'L',9:'K',10:'I',11:'H'}
        for i, average_ratio in enumerate(average_ratio_per_label):
            print(f"Label {label_dict[i]}: {average_ratio:.4f}")

    main(args.graph_path, args.model, args)
