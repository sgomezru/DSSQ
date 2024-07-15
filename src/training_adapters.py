import os
import sys
import torch
import torch.nn as nn

REPO_PATH = '/workspace/repositories/DSSQ/src'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(REPO_PATH)

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from models import get_unet
from data_utils import get_eval_data
from adapters import DimReductAdapter, DimReductModuleWrapper
from torch.utils.data import DataLoader

### Load basic config
DATA_KEY = 'prostate'
ITERATION = 0
LOG = False
AUGMENT = False
LOAD_ONLY_PRESENT = True
SUBSET = 'training' # Whether the validation is a subset or the whole set, normally bool, but for eval must be string 'training'
VALIDATION = True # If false makes validation set be the training one
EXTRA_DESCRIPTION = '_base'
N_DIMS = [2, 4, 8, 16, 32]

cfg = OmegaConf.load(f'{REPO_PATH}/configs/conf.yaml')
OmegaConf.update(cfg, 'run.iteration', ITERATION)
OmegaConf.update(cfg, 'run.data_key', DATA_KEY)

unet_name = 'monai-64-4-4'
cfg.wandb.log = LOG
cfg.wandb.project = f'{DATA_KEY}_{unet_name}_{ITERATION}{EXTRA_DESCRIPTION}'
cfg.format = 'numpy' # For eval (Adapter training is model on eval nonetheless)
args = unet_name.split('-')
cfg.unet[DATA_KEY].pre = unet_name
cfg.unet[DATA_KEY].arch = args[0]
cfg.unet[DATA_KEY].n_filters_init = None if unet_name == 'swinunetr' else int(args[1])
cfg.unet[DATA_KEY].training.augment = AUGMENT
cfg.unet[DATA_KEY].training.validation = VALIDATION
cfg.unet[DATA_KEY].training.subset = SUBSET
cfg.unet[DATA_KEY].training.load_only_present = LOAD_ONLY_PRESENT
cfg.unet[DATA_KEY].training.batch_size = 58 # Batch size hard coded based on dataset length and GPU capacity

if args[0] == 'monai':
    cfg.unet[DATA_KEY].depth = int(args[2])
    cfg.unet[DATA_KEY].num_res_units = int(args[3])

layer_names = [f'model.{"1.submodule." * i}0.conv' for i in range(cfg.unet[DATA_KEY].depth)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Loads the Siemens training dataset but in the required format for evaluation (No augmentations & numpy)
data = get_eval_data(train_set=False, val_set=False, eval_set=True, cfg=cfg)
dataset = data['eval']
dataloader = DataLoader(dataset, batch_size=cfg.unet[DATA_KEY].training.batch_size, shuffle=False, drop_last=False)
print(f'Length of dataset: {len(dataset)}')

dim_red_modes = ['IPCA', 'PCA']

for dim_red_mode in dim_red_modes:
    for n_dims in N_DIMS[::-1]:
        if dim_red_mode == 'PCA' and n_dims not in [2,4]:
            continue

        adapters = [DimReductAdapter(swivel, n_dims, cfg.unet[DATA_KEY].training.batch_size,
                                     mode=dim_red_mode, pre_fit=False, fit_gaussian=False,
                                     project=cfg.wandb.project) for swivel in layer_names]

        adapters = nn.ModuleList(adapters)
        unet, state_dict = get_unet(cfg, update_cfg_with_swivels=False, return_state_dict=True)
        unet_adapted = DimReductModuleWrapper(model=unet, adapters=adapters)
        unet_adapted.to(device);
        unet_adapted.eval();

        print(f'Training {dim_red_mode} module of {n_dims} dims')
        for i, batch in enumerate(tqdm(dataloader)):
            input_ = batch['input'].to(device)
            if input_.size(0) < n_dims and dim_red_mode == 'IPCA':
                print(f'Skipped because batch size smaller than n_components')
                continue
            unet_adapted(input_)
        if dim_red_mode == 'PCA': unet_adapted.fit_adapters_modules()
        unet_adapted.save_adapters_modules()
        unet_adapted.set_pre_fit()
        unet_adapted.set_fit_gaussian()
        print(f'Fitting gaussian for {dim_red_mode} module of {n_dims} dims')
        for i, batch in enumerate(tqdm(dataloader)):
            input_ = batch['input'].to(device)
            unet_adapted(input_)
        unet_adapted.fit_adapters_gaussians()
