from tqdm import tqdm
import torch
import wandb

def train_loop(model, train_loader, val_loader, optimizer, criterion, device, cfg, log=False):
    '''
    Return train_loss_list, val_loss_list, train_acc_list, val_acc_list
    '''
    data_key = cfg['data_key']
    seg_key = cfg['seg_key']
    stats = {
        'train_loss_list':  [],
        'val_loss_list' : [],
        'train_acc_list' : [],
        'val_acc_list' : []
    }
    for epoch in tqdm(range(cfg['num_epochs']), desc="Epochs", position=0):
        model.train()
        train_loss = 0
        correct_pixels = 0
        num_pixels = 0
        for batch in tqdm(train_loader, desc="Train batches", leave=False, position=1):
            optimizer.zero_grad()
            x = batch[data_key].float().to(device)
            y = batch[seg_key].squeeze(1).to(device)
            out = model(x)
            pred = torch.argmax(out.detach(), dim = 1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
            correct_pixels += (pred == y).sum().item()
            num_pixels += y.nelement()
        train_loss /= len(train_loader.dataset)
        stats['train_loss_list'].append(train_loss)
        train_acc = correct_pixels / num_pixels
        stats['train_acc_list'].append(train_acc)
        model.eval()
        val_loss = 0
        correct_pixels = 0
        num_pixels = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val batches", leave=False, position=1):
                x = batch[data_key].float().to(device)
                y = batch[seg_key].squeeze(1).to(device)
                out = model(x)
                pred = torch.argmax(out, dim = 1)
                loss = criterion(out, y)
                val_loss += loss.item() * x.shape[0]
                correct_pixels += (pred == y).sum().item()
                num_pixels += y.nelement()
            val_loss /= len(val_loader.dataset)
            stats['val_loss_list'].append(val_loss)
            val_acc = correct_pixels / num_pixels
            stats['val_acc_list'].append(val_acc)
        if log:
            wandb.log({'train_acc': train_acc, 'val_acc': val_acc,
                        'train_loss': train_loss, 'val_loss': val_loss})
    return stats
