import os
import sys
from omegaconf import OmegaConf
import wandb

REPO_PATH = '/workspace/repositories/DSSQ/src'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(REPO_PATH)

from models import get_unet
from data_utils import get_pmri_data_loaders
from unet_trainer import get_unet_trainer

### Load basic config
DATA_KEY = 'prostate'
ITERATION = 0
AUGMENT = True
SUBSET = False # Whether the validation is a subset or the whole set
VALIDATION = True # If false makes validation set be the training one
EXTRA_DESCRIPTION = '_base'
LOAD_ONLY_PRESENT = True
cfg = OmegaConf.load(f'{REPO_PATH}/configs/conf.yaml')
OmegaConf.update(cfg, 'run.iteration', ITERATION)
OmegaConf.update(cfg, 'run.data_key', DATA_KEY)

unet_name = 'monai-64-4-4'
cfg.wandb.project = f'{DATA_KEY}_{unet_name}_{ITERATION}{EXTRA_DESCRIPTION}'
args = unet_name.split('-')
cfg.unet[DATA_KEY].pre = unet_name
cfg.unet[DATA_KEY].arch = args[0]
cfg.unet[DATA_KEY].n_filters_init = None if unet_name == 'swinunetr' else int(args[1])
cfg.unet[DATA_KEY].training.augment = AUGMENT
cfg.unet[DATA_KEY].training.load_only_present = LOAD_ONLY_PRESENT
cfg.unet[DATA_KEY].training.validation = VALIDATION
cfg.unet[DATA_KEY].training.subset = SUBSET
cfg.format = 'torch'

if args[0] == 'monai':
    cfg.unet[DATA_KEY].depth = int(args[2])
    cfg.unet[DATA_KEY].num_res_units = int(args[3])

wandb.init(
    project=cfg.wandb.project,
    config={
        "learning_rate": cfg.unet[DATA_KEY].training.lr,
        "architecture": unet_name,
        "dataset": DATA_KEY
    }
)

unet = get_unet(cfg, return_state_dict=False)
train_loader, val_loader = get_pmri_data_loaders(cfg=cfg)
pmri_trainer = get_unet_trainer(cfg=cfg, train_loader=train_loader, val_loader=val_loader, model=unet)

try:
    pmri_trainer.fit()
finally:
    wandb.finish()
