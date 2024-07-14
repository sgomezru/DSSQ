import os
import sys
import pickle
import torch
import torch.nn as nn

REPO_PATH = '/workspace/repositories/DSSQ/src'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(REPO_PATH)

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from models import get_unet
from data_utils import get_eval_data
from adapters import PCA_Adapter, PCAModuleWrapper
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
cfg.unet[DATA_KEY].training.batch_size = 58

if args[0] == 'monai':
    cfg.unet[DATA_KEY].depth = int(args[2])
    cfg.unet[DATA_KEY].num_res_units = int(args[3])

layer_names = [f'model.{"1.submodule." * i}0.conv' for i in range(cfg.unet[DATA_KEY].depth)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Loads the Siemens training dataset but in the required format for evaluation (No augmentations & numpy)
data = get_eval_data(train_set=False, val_set=False, eval_set=True, cfg=cfg)
dataset = data['eval']
print(f'Length of dataset: {len(dataset)}')

possible_modes = ['train_adapters', 'get_activations']

for mode in possible_modes:
    print(f'Running on mode: {mode}')
    for n_dims in N_DIMS[::-1]:
        # Batch size hard coded based on dataset length and GPU capacity
        pre_fit, train_gaussian = None, None
        if mode == 'train_adapters':
            pre_fit, train_gaussian, name = False, False, ''
        elif mode == 'get_activations':
            pre_fit, train_gaussian, name = True, True, cfg.wandb.project

        adapters = [PCA_Adapter(swivel, n_dims, cfg.unet[DATA_KEY].training.batch_size,
                                pre_fit, train_gaussian, name=name) for swivel in layer_names]

        adapters = nn.ModuleList(adapters)
        unet, state_dict = get_unet(cfg, update_cfg_with_swivels=False, return_state_dict=True)
        unet_adapted = PCAModuleWrapper(model=unet, adapters=adapters)
        unet_adapted.to(device);
        unet_adapted.eval();

        dataloader = DataLoader(dataset, batch_size=cfg.unet[DATA_KEY].training.batch_size,
                                shuffle=False, drop_last=False)

        if mode == 'train_adapters':
            print(f'Training for PCA with {n_dims} dims')
            for i, batch in enumerate(tqdm(dataloader)):
                input_ = batch['input'].to(device)
                if input_.size(0) < n_dims:
                    print(f'Skipped because batch size smaller than n_components')
                    continue
                unet_adapted(input_)

            for adapter in unet_adapted.adapters:
                name = adapter.swivel.replace('.', '_')
                with open(f'/workspace/out/pca/{cfg.wandb.project}_{name}_{n_dims}dim.pkl',  'wb') as f:
                    pickle.dump(adapter.pca, f)

        elif mode == 'get_activations':
            print(f'Getting activations for adapter with {n_dims}')
            for i, batch in enumerate(tqdm(dataloader)):
                input_ = batch['input'].to(device)
                unet_adapted(input_)

            for adapter in unet_adapted.adapters:
                adapter._save_activations_np()
