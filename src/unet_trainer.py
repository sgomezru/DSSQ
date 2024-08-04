## UNet Trainer

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from typing import Dict, Callable
import time
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm
import numpy as np
from utils import EarlyStopping, epoch_average
from monai.losses.dice import DiceCELoss


def get_unet_trainer(cfg, train_loader, val_loader, model):
    """Wrapper function to instantiate a unet trainer for either ACDC or
    Calgary-Campinas dataset

    Args:
        cfg (OmegaConf): general config with segmentation model information
            Contains the task-specific base config for the model and:
                log
                debug
                weight_dir
                log_dir
                data_key
                iteration
        train_loader (DataLoader): training data loader
        val_loader (DataLoader): validation data loader
        model (nn.Module): (unet) model

    Returns:
        UNetTrainerACDC or UNetTrainerCalgary: trainer object
    """
    if cfg.run.data_key == "prostate":
        trainer = get_unet_prostate_trainer(cfg, train_loader, val_loader, model)
    else:
        raise NotImplementedError

    return trainer


def get_unet_prostate_trainer(cfg, train_loader, val_loader, model):
    """
    Trainer for Multisite PMRI dataset
    """
    model_cfg = cfg.unet[cfg.run.data_key]
    num_batches_per_epoch = model_cfg.training.num_batches_per_epoch
    num_val_batches_per_epoch = model_cfg.training.num_val_batches_per_epoch
    name = f"{cfg.run.data_key}_{model_cfg.pre}_{cfg.wandb.project}_{cfg.run.iteration}"
    weight_dir = (cfg.fs.weight_dir,)
    log_dir = (cfg.fs.log_dir,)
    lr = model_cfg.training.lr
    n_epochs = model_cfg.training.epochs
    patience = model_cfg.training.patience
    log = cfg.wandb.log
    criterion = DiceCELoss(softmax=True, to_onehot_y=True)

    return UNetTrainerPMRI(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        num_batches_per_epoch=num_batches_per_epoch,
        num_val_batches_per_epoch=num_val_batches_per_epoch,
        weight_dir=weight_dir,
        log_dir=log_dir,
        lr=lr,
        n_epochs=n_epochs,
        patience=patience,
        es_mode="min",
        eval_metrics=None,
        log=log,
        name=name,
    )


class UNetTrainerPMRI:
    """
    Trainer class for training and evaluating a U-Net model for PMRI dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_batches_per_epoch,
        num_val_batches_per_epoch,
        weight_dir: str,
        log_dir: str,
        lr: float = 1e-4,
        n_epochs: int = 250,
        patience: int = 5,
        es_mode: str = "min",
        eval_metrics: Dict[str, nn.Module] = None,
        log: bool = True,
        name: str = "pmri-unet",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.model = model.to(self.device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_val_batches_per_epoch = num_val_batches_per_epoch
        self.weight_dir = weight_dir[0] if isinstance(weight_dir, tuple) else weight_dir
        self.log_dir = log_dir[0] if isinstance(log_dir, tuple) else log_dir
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.es_mode = es_mode
        self.eval_metrics = eval_metrics
        self.log = log
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=self.patience
        )
        self.es = EarlyStopping(mode=self.es_mode, patience=2 * self.patience)
        self.scaler = GradScaler()
        self.history = {"train loss": [], "valid loss": []}
        self.training_time = 0
        if self.eval_metrics is not None:
            self.history = {
                **self.history,
                **{key: [] for key in self.eval_metrics.keys()},
            }
        if self.log:
            wandb.watch(self.model)

    def inference_step(self, x):
        return self.model(x.to(self.device))

    def save_hist(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        savepath = os.path.join(self.log_dir, f"{self.name}.npy")
        np.save(savepath, self.history)
        return

    def save_model(self, epoch):
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        savepath = os.path.join(self.weight_dir, f"{self.name}_best.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            savepath,
        )
        self.save_hist()
        return

    def load_model(self):
        savepath = os.path.join(self.weight_dir, f"{self.name}_best.pt")
        checkpoint = torch.load(savepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        savepath = os.path.join(self.log_dir, f"{self.name}.npy")
        self.history = np.load(savepath, allow_pickle="TRUE").item()
        print("Loaded model and optimizer")
        return

    def train_epoch(self):
        loss_list, batch_sizes = [], []
        for it in range(self.num_batches_per_epoch):
            batch = next(self.train_loader)
            input_ = batch["data"].float()
            target = batch["target"].to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=False):
                net_out = self.inference_step(input_)
                loss = self.criterion(net_out, target)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0, norm_type=2.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_list.append(loss.item())
            batch_sizes.append(input_.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history["train loss"].append(average_loss)

        if self.log:
            wandb.log({"train_loss": average_loss}, commit=False)

        return average_loss

    def get_sample(self, mode: str = "valid"):
        if mode == "valid":
            self.model.eval()
            data, target, _ = next(iter(self.val_loader)).values()
            net_out = self.inference_step(data)
            self.model.train()
        else:
            data, target, _ = next(iter(self.train_loader)).values()
            net_out = self.inference_step(data)
        x_hat = net_out

        return data.cpu(), target.cpu(), x_hat.cpu()

    @torch.no_grad()
    def eval_epoch(self):
        loss_list, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for it in range(self.num_val_batches_per_epoch):
            batch = next(self.val_loader)
            input_ = batch["data"].float()
            target = batch["target"].to(self.device)
            net_out = self.inference_step(input_)
            loss = self.criterion(net_out, target)
            loss_list.append(loss.item())
            batch_sizes.append(input_.shape[0])
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    epoch_metrics[key].append(
                        metric(net_out, target).detach().mean().cpu()
                    )
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history["valid loss"].append(average_loss)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                avrg = epoch_average(epoch_scores, batch_sizes)
                self.history[key].append(avrg)
                if self.log:
                    wandb.log({key: avrg}, commit=False)

        if self.log:
            wandb.log(
                {
                    "valid_loss": average_loss,
                },
                commit=False,
            )

        return average_loss

    @torch.no_grad()
    def test_set(self, testloader: DataLoader) -> dict:
        self.model.eval()

        metric, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}
        for batch in testloader:
            input_ = batch["data"].float()
            target = batch["target"].to(self.device)
            batch_sizes.append(input_.shape[0])

            input_chunks = torch.split(input_, 32, dim=0)
            target_chunks = torch.split(target, 32, dim=0)
            net_out = []
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                net_out_chunk = self.inference_step(input_chunk)
                net_out.append(net_out_chunk.detach().cpu())

            net_out = torch.cat(net_out, dim=0)
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    epoch_metrics[key].append(
                        metric(net_out, target).detach().mean().cpu()
                    )

        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)

        return epoch_metrics

    def fit(self):
        best_es_metric = 1e25 if self.es_mode == "min" else -1e25
        progress_bar = tqdm(
            range(self.n_epochs), total=self.n_epochs, position=0, leave=True
        )
        self.model.eval()
        valid_loss = self.eval_epoch()
        self.training_time = time.time()

        if self.log:
            wandb.log({}, commit=True)

        for epoch in progress_bar:
            self.model.train()
            train_loss = self.train_epoch()
            self.model.eval()
            valid_loss = self.eval_epoch()
            self.scheduler.step(valid_loss)

            epoch_summary = (
                [f"Epoch {epoch+1}"]
                + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history]
                + [f"ES epochs: {self.es.num_bad_epochs}"]
            )
            progress_bar.set_description("".join(epoch_summary))
            es_metric = list(self.history.values())[1][-1]

            if self.log:
                wandb.log({}, commit=True)

            if self.es_mode == "min":
                if es_metric < best_es_metric:
                    best_es_metric = es_metric
                    self.save_model(epoch)
            else:
                if es_metric > best_es_metric:
                    best_es_metric = es_metric
                    self.save_model(epoch)
            if self.es.step(es_metric):
                print("Early stopping triggered!")
                break

        self.training_time = time.time() - self.training_time
        self.save_hist()
        self.load_model()
        print(f"Total training time (min): {self.training_time / 60.}")

    def fit_adapter(self):
        progress_bar = tqdm(
            range(self.num_batches_per_epoch),
            total=self.num_batches_per_epoch,
            position=0,
            leave=True,
        )
        self.model.eval()
        self.training_time = time.time()
        with torch.no_grad():
            for it in progress_bar:
                batch = next(self.train_loader)
                input_ = batch["data"].float()
                self.inference_step(input_)
        self.training_time = time.time() - self.training_time
        print(f"Total training time (min): {self.training_time / 60.}")
