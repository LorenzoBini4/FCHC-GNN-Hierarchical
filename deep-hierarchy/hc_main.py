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
from model.hc_model import *

# Set seeds for reproducibility
seed_value = 77
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
R = R.unsqueeze(0).to(device)

# One training epoch for GNN model.
with ClearCache():
    def train(train_loader, model, optimizer, device, criterion):
        model.train()
        for data in tqdm(train_loader, desc='Training'):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # MCLoss
            constr_output = get_constr_out(output, R)
            train_output = data.y * output.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1 - data.y) * constr_output.double() + data.y * train_output
            loss = criterion(train_output[:, data.to_eval[0]], data.y[:, data.to_eval[0]])
            loss.backward()
            optimizer.step()

# Get acc. of GNN model.
with ClearCache():
    with torch.no_grad():
        def val(loader, model, device):
            model.eval()
            correct = 0
            for data in tqdm(loader, desc='Validation'):
                data = data.to(device)
                constrained_output = model(data)
                predss = constrained_output.data > 0.4
                correct += (predss == data.y.byte()).sum() / (predss.shape[0] * predss.shape[1])
            return correct / len(loader.dataset)

def gnn_evaluation(gnn, max_num_epochs, batch_size, start_lr, num_repetitions, min_lr=0.000001, factor=0.05, patience=7, all_std=True):
    dataset = MyGraphDataset(num_samples=len(torch.load(INPUT_GRAPH))).shuffle()
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

            model = gnn(R).to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr) #weight_decay=0.0001
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                                   patience=patience, min_lr=0.0000001)
            criterion = nn.BCELoss()
            best_val_acc = 0.0
            early_stopping_counter = 0
            early_stopping_patience = 20

            best_model = None
            best_val_acc = 0.0
            for epoch in range(1, max_num_epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']
                torch.cuda.empty_cache()
                train(train_loader, model, optimizer, device, criterion)
                torch.cuda.empty_cache()
                val_acc = val(val_loader, model, device)
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

            torch.save(best_model, 'best_gat.pt')
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

max_num_epochs=200
batch_size=1
start_lr=0.001
num_repetitions=4
patient_dict=gnn_evaluation(HCGAT, max_num_epochs, batch_size, start_lr, num_repetitions, all_std=True)

# Initialize a list to store the ratios for each label across all patients
average_ratio_per_label = []
F_scores = np.zeros((19,num_repetitions))
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
