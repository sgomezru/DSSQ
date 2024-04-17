import numpy as np
import os
import sys
import torch
import wandb
from monai.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from torch import optim

base_path = Path('/workspace/src').resolve()
src_path = base_path / 'src'
sys.path.append(str(src_path))

from datasets import load_dataset
from data_utils import Transform
from models import get_model

def train_loop(model, train_loader, val_loader, optimizer, criterion, device, cfg, log=False):
    '''
    Return train_loss_list, val_loss_list, train_acc_list, val_acc_list
    '''
    dataset_key = cfg.run.dataset_key
    dataset_subkey = cfg.run.dataset_subkey
    arch = cfg.run.arch
    model_details = cfg[arch.split('-')[1]][dataset_key]
    if log:
        wandb.init(
            project=cfg.wandb.project,
            config={
                "learning_rate": model_details.training.lr,
                "architecture": arch,
                "dataset": dataset_key + '-' + dataset_subkey,
                "epochs": model_details.training.epochs,
                "batch_size": model_details.training.batch_size,
                "batches_per_epoch": model_details.training.num_batches_per_epoch
            }
        )

    data_key = cfg.data[dataset_key][dataset_subkey].data_key
    seg_key = cfg.data[dataset_key][dataset_subkey].seg_key
    stats = {
        'train_loss_list':  [],
        'val_loss_list' : [],
        'train_acc_list' : [],
        'val_acc_list' : []
    }
    for epoch in tqdm(range(model_details.training.epochs), desc="Epochs", position=0):
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
    if log:
        wandb.finish()

    return stats

class TrainerManager(object):

    def __init__(self, cfg):
        self.dataset_key = cfg.run.dataset_key
        self.dataset_subkey = cfg.run.dataset_subkey
        self.arch = cfg.run.arch
        self.model_name = cfg.run.arch.split('-')[1]
        self.name = f'{cfg.wandb.project}_{self.dataset_key}_{self.dataset_subkey}_{self.arch}'
        self.weight_dir = Path(cfg.fs.weight_dir).resolve()
        self.log_dir = Path(cfg.fs.log_dir).resolve()
        self.training_config = cfg[self.arch.split('-')[1]][self.dataset_key][self.dataset_subkey]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(cfg)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_config.training.lr)
        self.history = { 'train loss': [], 'valid loss': [] }
        self.transforms = Transform(cfg)
        if cfg.run.load is True: self.load_model(cfg);
        if cfg.run.load_dataset is True: self.load_dataloaders(cfg);
        if cfg.wandb.log is True: self.init_logging(cfg);

    def init_logging(self, cfg):
        wandb.init(
            project=cfg.wandb.project,
            config={
                "learning_rate": self.training_config.training.lr,
                "architecture": self.arch,
                "dataset": self.dataset_key + '-' + self.dataset_subkey,
                "epochs": self.training_config.training.epochs,
                "batch_size": self.training_config.training.batch_size,
                "batches_per_epoch": self.training_config.training.num_batches_per_epoch
            }
        )

    def load_dataloaders(self, cfg):
        print('Loading datasets...')
        data = load_dataset(cfg)
        train_dataset = Dataset(data['train'], transform=self.transforms['all_transforms'])
        valid_dataset = Dataset(data['valid'], transform=self.transforms['base_transforms'])
        self.train_loader = DataLoader(train_dataset, shuffle=True,
                                       batch_size=self.training_config.training.batch_size)
        self.valid_loader = DataLoader(valid_dataset, shuffle=True,
                                       batch_size=self.training_config.training.batch_size)
        self.data_key = cfg.data[self.dataset_key][self.dataset_subkey].data_key
        self.seg_key = cfg.data[self.dataset_key][self.dataset_subkey].seg_key
    
    def fit(self):
        for epoch in tqdm(range(self.training_config.training.epochs), desc="Epochs", position=0):
            self.model.train()
            train_loss = 0
            correct_pixels = 0
            num_pixels = 0
            for batch in tqdm(self.train_loader, desc="Train batches", leave=False, position=1):
                self.optimizer.zero_grad()
                x = batch[self.data_key].float().to(self.device)
                y = batch[self.seg_key].squeeze(1).to(self.device)
                out = self.model(x)
                pred = torch.argmax(out.detach(), dim = 1)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x.shape[0]
                correct_pixels += (pred == y).sum().item()
                num_pixels += y.nelement()
            train_loss /= len(self.train_loader.dataset)
            self.history['train_loss'].append(train_loss)
            train_acc = correct_pixels / num_pixels
            self.history['train_acc'].append(train_acc)
            self.model.eval()
            val_loss = 0
            correct_pixels = 0
            num_pixels = 0
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Val batches", leave=False, position=1):
                    x = batch[self.data_key].float().to(self.device)
                    y = batch[self.seg_key].squeeze(1).to(self.device)
                    out = self.model(x)
                    pred = torch.argmax(out, dim = 1)
                    loss = self.criterion(out, y)
                    val_loss += loss.item() * x.shape[0]
                    correct_pixels += (pred == y).sum().item()
                    num_pixels += y.nelement()
                val_loss /= len(self.val_loader.dataset)
                self.history['val_loss'].append(val_loss)
                val_acc = correct_pixels / num_pixels
                self.history['val_acc'].append(val_acc)
            if self.log:
                wandb.log({'train_acc': train_acc, 'val_acc': val_acc,
                            'train_loss': train_loss, 'val_loss': val_loss})
        # if self.log:
        #     wandb.finish()
    print(f'Finished training')

    def eval(self):
        pass

    def save_stats_history(self):
        if(not os.path.exists(self.log_dir)):
            os.makedirs(self.log_dir)
        savepath = f'{self.log_dir}{self.name}.npy'
        np.save(savepath, self.history)
        print(f'History stats of model saved')
    
    def save_model(self):
        if(not os.path.exists(self.weight_dir)):
            os.makedirs(self.weight_dir)
        savepath = f'{self.weight_dir}{self.name}_best.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, savepath)
        self.save_stats_history()
        print(f'Model saved')
    
    def load_model(self):
        savepath = f'{self.weight_dir}{self.name}_best.pt'
        checkpoint = torch.load(savepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        savepath = f'{self.log_dir}{self.name}.npy'
        self.history = np.load(savepath, allow_pickle='TRUE').item()
        print(f'Model and optimizer params loaded, as well as stats history')
