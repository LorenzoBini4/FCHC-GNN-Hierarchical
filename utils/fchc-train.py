import torch

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.empty_cache()
        
# One training epoch for the FCHC-GNN plug-in module.
with ClearCache():
    def train_deep(train_loader, model, optimizer, device, criterion):
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

# Get acc. of FCHC-GNN model.
with ClearCache():
    with torch.no_grad():
        def val_deep(loader, model, device):
            model.eval()
            correct = 0
            for data in tqdm(loader, desc='Validation'):
                data = data.to(device)
                constrained_output = model(data)
                predss = constrained_output.data > 0.4
                correct += (predss == data.y.byte()).sum() / (predss.shape[0] * predss.shape[1])
            return correct / len(loader.dataset)

with ClearCache():
    # One training epoch for the FCHC-GNN model.
    def train_shallow(train_loader, model, optimizer, device):
        model.train()
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            criterion = nn.NLLLoss()
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': loss.item()})

    # Get acc. of FCHC-GNN model.
    def val_shallow(loader, model, device):
        with torch.no_grad():
            model.eval()
            correct = 0
            pbar = tqdm(loader, desc="Validation")
            for data in pbar:
                data = data.to(device)
                output = model(data)
                pred = output.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()/len(pred)
                pbar.set_postfix({'Accuracy': correct / len(loader.dataset)})
            return correct / len(loader.dataset)
