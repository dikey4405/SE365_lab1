import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os

from model import build_model
from dataloader import get_dataloaders, collate_fn
from utils import set_seed

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        if batch is None: continue
            
        inputs = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            
            inputs = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total = len(all_labels)
    epoch_loss = running_loss / total if total > 0 else 0.0
    
    metrics = {
        'loss': epoch_loss,
        'acc': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'preds': all_preds,
        'labels': all_labels
    }
    return metrics

def run_training(config, data_dir):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Exp: {config['name']} on {device} ---")
    
    train_set, val_set, test_set = get_dataloaders(
        data_dir, 
        image_size=128, 
        batch_size=config['batch_size'],
        use_augmentation=config.get('augmentation', False)
    )
    
    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, 
                             num_workers=num_workers, collate_fn=collate_fn)

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    
    lr = config.get('lr', 1e-3)
    wd = config.get('weight_decay', 0.0)
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = config.get('patience', 5)
    counter = 0
    
    for epoch in range(config['epochs']):
        start = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        v_loss, v_acc = val_metrics['loss'], val_metrics['acc']
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {epoch+1:02d} | T_Loss: {t_loss:.4f} T_Acc: {t_acc:.4f} | "
              f"V_Loss: {v_loss:.4f} V_Acc: {v_acc:.4f} | Time: {time.time()-start:.1f}s")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            counter = 0
            torch.save(model.state_dict(), f"best_model_{config['name']}.pth")
        else:
            counter += 1
            if config.get('early_stopping', False) and counter >= patience:
                print("Early stopping triggered!")
                break
                
    model.load_state_dict(torch.load(f"best_model_{config['name']}.pth", weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    return history, test_metrics, ['Cat', 'Dog']