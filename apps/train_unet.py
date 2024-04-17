import sys
from pathlib import Path
from omegaconf import OmegaConf

base_path = Path('/workspace/src').resolve()
src_path = base_path / 'src'
cfg_path = base_path / 'config'
pmri_data_path = Path('/data/Data/PMRI').resolve()
sys.path.append(str(src_path))
cfg = OmegaConf.load(cfg_path / 'conf.yaml')

DATASET_KEY = 'prostate'
DATASET_SUBKEY = 'pmri'
ARCH = 'monai-unet-64-4-4'
PROJECT_NAME = 'test-UNet'
VALIDATION = True
OmegaConf.update(cfg, 'run.dataset_key', DATASET_KEY)
OmegaConf.update(cfg, 'run.dataset_subkey', DATASET_SUBKEY)
OmegaConf.update(cfg, 'run.arch', ARCH)
OmegaConf.update(cfg, 'run.validation', VALIDATION)
OmegaConf.update(cfg, 'wandb.project', PROJECT_NAME)

import torch
from torch import optim, nn
from monai.data import DataLoader, Dataset

from datasets import load_dataset
from data_utils import Transform
from trainer import train_loop
from models import get_model

model_type = cfg.run.arch.split('-')[1]
data = load_dataset(cfg)
ds_train = data['train']
ds_valid = data['valid']

transforms = Transform(cfg)
batch_size = cfg[model_type][DATASET_KEY].training.batch_size
lr = cfg[model_type][DATASET_KEY].training.lr
dst = Dataset(ds_train, transform=transforms['all_transforms'])
dsv = Dataset(ds_valid, transform=transforms['base_transforms'])
dlt = DataLoader(dst, shuffle=True, batch_size=batch_size)
dlv = DataLoader(dsv, batch_size=batch_size)

device = torch.device('cuda:3')
unet = get_model(cfg)
unet.to(device)
optimizer = optim.Adam(unet.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

training_stats = train_loop(
    model=unet,
    train_loader=dlt,
    val_loader=dlv,
    optimizer=optimizer,
    criterion=loss_function,
    device=device,
    cfg=cfg,
    log=True
)
