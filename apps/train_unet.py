import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

base_path = Path('/workspace/src').resolve()
src_path = base_path / 'src'
cfg_path = base_path / 'config'
pmri_data_path = Path('/data/Data/PMRI').resolve()
sys.path.append(str(src_path))

from trainer import TrainerManager

#####################CONFIG LOADING########################################
DATASET_KEY = 'prostate'
DATASET_SUBKEY = 'pmri'
ARCH = 'monai-unet-64-4-4'
PROJECT_NAME = 'micro'
VALIDATION = True
LOG = True
LOAD_MODEL = False
LOAD_DATASETS = True
NUM_EPOCHS = 80
BATCH_SIZE = 32
cfg = OmegaConf.load(cfg_path / 'conf.yaml')
OmegaConf.update(cfg, 'run.dataset_key', DATASET_KEY)
OmegaConf.update(cfg, 'run.dataset_subkey', DATASET_SUBKEY)
OmegaConf.update(cfg, 'run.arch', ARCH)
OmegaConf.update(cfg, 'run.validation', VALIDATION)
OmegaConf.update(cfg, 'wandb.project', PROJECT_NAME)
OmegaConf.update(cfg, 'wandb.log', LOG)
OmegaConf.update(cfg, f'unet.{DATASET_KEY}.{DATASET_SUBKEY}.training.epochs', NUM_EPOCHS)
OmegaConf.update(cfg, f'unet.{DATASET_KEY}.{DATASET_SUBKEY}.training.batch_size', BATCH_SIZE)
### If want to load a pre-trained model
OmegaConf.update(cfg, 'run.load', LOAD_MODEL)
### If want to load the datasets
OmegaConf.update(cfg, 'run.load_dataset', LOAD_DATASETS)
###########################################################################

trainer = TrainerManager(cfg, eval_metrics = {'train_acc': 0, 'valid_acc': 1})

trainer.fit()
trainer.stop_logging()
