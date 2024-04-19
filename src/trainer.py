import numpy as np
import os
import sys
import torch
import wandb
from monai.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from torch import optim, nn
from typing import Dict

base_path = Path('/workspace/src').resolve()
src_path = base_path / 'src'
sys.path.append(str(src_path))

from datasets import load_dataset
from data_utils import Transform
from models import get_model

class TrainerManager(object):

    def __init__(self, cfg, criterion = nn.CrossEntropyLoss(), eval_metrics : Dict[str, nn.Module] = None):
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
        self.criterion = criterion
        self.history = { 'train_loss': [], 'valid_loss': [] }
        self.eval_metrics = eval_metrics
        self.transforms = Transform(cfg)
        if self.eval_metrics is not None:
            self.history = {**self.history, **{metric: [] for metric in self.eval_metrics.keys()}}
        if cfg.run.get('load') is True: self.load_model();
        if cfg.run.get('load_dataset') is True: self.load_dataloaders(cfg);
        if cfg.wandb.log is True: self.init_logging(cfg);

    def init_logging(self, cfg):
        self.log = True
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

    def stop_logging(self):
        wandb.finish()

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
        best_train_loss = np.inf
        best_valid_loss = np.inf
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
            valid_loss = 0
            correct_pixels = 0
            num_pixels = 0
            with torch.no_grad():
                for batch in tqdm(self.valid_loader, desc="Validation batches", leave=False, position=1):
                    x = batch[self.data_key].float().to(self.device)
                    y = batch[self.seg_key].squeeze(1).to(self.device)
                    out = self.model(x)
                    pred = torch.argmax(out, dim = 1)
                    loss = self.criterion(out, y)
                    valid_loss += loss.item() * x.shape[0]
                    correct_pixels += (pred == y).sum().item()
                    num_pixels += y.nelement()
                valid_loss /= len(self.valid_loader.dataset)
                self.history['valid_loss'].append(valid_loss)
                valid_acc = correct_pixels / num_pixels
                self.history['valid_acc'].append(valid_acc)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                self.save_model(epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model(epoch)
            if self.log:
                wandb.log({'train_acc': train_acc, 'valid_acc': valid_acc,
                            'train_loss': train_loss, 'valid_loss': valid_loss})
        print(f'Finished training')

    def eval(self):
        pass

    def save_stats_history(self):
        if(not os.path.exists(self.log_dir)):
            os.makedirs(self.log_dir)
        savepath = self.log_dir / f'{self.name}.npy'
        np.save(savepath, self.history)
        print(f'History stats of model saved')
    
    def save_model(self, epoch):
        if(not os.path.exists(self.weight_dir)):
            os.makedirs(self.weight_dir)
        savepath = self.weight_dir / f'{self.name}_best.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }, savepath)
        self.save_stats_history()
        print(f'Model saved')
    
    def load_model(self):
        savepath = self.weight_dir / f'{self.name}_best.pt'
        checkpoint = torch.load(savepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        savepath = self.log_dir / f'{self.name}.npy'
        self.history = np.load(savepath, allow_pickle='TRUE').item()
        print(f'Model and optimizer params loaded, as well as stats history')
