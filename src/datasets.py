import nibabel as nib
import numpy as np
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset

class MultisiteMRIProstateDataset(Dataset):
    '''
    Multi-site dataset for prostate MRI segmentation from:
    https://liuquande.github.io/SAML/
    Possible vendors:
    - Siemens
    - GE
    - Philips
    Initialization parameters:
    - datapath -> Path to dataset directory
    - vendor -> Vendor from possible ones to load data
    - split -> Either all, train, test, result depends on seed, it's done
    80/20 based on the number of cases (Note that it isn't the same as
    number of total slices)
    - load_only_present -> Defaults to False, set to True if desired to
    load only the slices which contain the prostate class in their
    segmentation mask (i.e. omit the ones that are only "background")
    - seed -> Random seed which defaults to 42
    '''
    _VENDORS_INFO = {
        'RUNMC': ['A', 'siemens'],
        'BMC':   ['B', 'philips'],
        'I2CVB': ['C', 'siemens'],
        'UCL':   ['D', 'siemens'],
        'BIDMC': ['E', 'ge'],
        'HK':    ['F', 'siemens']
    }
    _DS_CONFIG = {
        'num_classes': 2,
        'spatial_dims': 2,
        'size': (384, 384) 
    }

    def __init__(self, datapath, vendor, split='all', load_only_present=False, seed=42):
        assert vendor in ['siemens', 'ge', 'philips'], 'Invalid vendor'
        assert split in ['all', 'train', 'valid'], 'Invalid split'
        self.vendor = vendor
        self._datapath = Path(datapath).resolve()
        self._split = split
        self._load_only_present = load_only_present
        self._seed = seed
        self._load_data()

    def _load_data(self):
        print("Loading dataset...")
        self.input = []
        self.target = []
        vendor_sites = [site for site, info in self._VENDORS_INFO.items()
                        if info[-1] == self.vendor]
        for site in vendor_sites:
            site_path = self._datapath / site
            for file in site_path.iterdir():
                # Load the ones that have a segmentation associated file to them
                if 'segmentation' in file.name.lower():
                    case = file.name[4:6]
                    seg_name = 'Segmentation' if site == 'BMC' else 'segmentation'
                    case_input_path = site_path / f'Case{case}.nii.gz'
                    case_target_path = site_path / f'Case{case}_{seg_name}.nii.gz'
                    x = nib.load(case_input_path)
                    y = nib.load(case_target_path)
                    x = x.get_fdata()
                    y = y.get_fdata().astype(int)
                    if self._load_only_present:
                        case_valid_indices = []
                        for slice_idx in range(y.shape[-1]):
                            if len(np.unique(y[..., slice_idx])) > 1:
                                case_valid_indices.append(slice_idx)
                        x = x[..., case_valid_indices]
                        y = y[..., case_valid_indices]
                    self.input.append(x)
                    self.target.append(y)
        # Split here
        if self._split != 'all':
            num_cases = list(range(len(self.input)))
            rng = random.Random(self._seed)
            rng.shuffle(num_cases)
            split_idx = int(len(num_cases) * 0.8)
            cases = num_cases[:split_idx] if self._split == 'train' else num_cases[split_idx:]
            self.input = [self.input[i] for i in cases]
            
        # Concat around last axis (single slices)
        self.input = torch.from_numpy(np.concatenate(self.input, axis=-1))
        self.target = torch.from_numpy(np.concatenate(self.target, axis=-1))
        # Relabel cases if there are two prostate classes (Since not all datasets distinguish between the two)
        self.target[self.target == 2] = 1
        # Adding channel dimension
        self.input = self.input.unsqueeze(0)
        self.target = self.target.unsqueeze(0)

    def __len__(self):
        return self.input.shape[-1]

    def __getitem__(self, idx):
        return {
            "image": self.input[..., idx],
            "label": self.target[..., idx]
        }

def load_dataset(cfg):
    dataset_key = cfg.run.dataset_key
    dataset_subkey = cfg.run.dataset_subkey
    model_arch = cfg.run.arch
    net = model_arch.split('-')[1]
    data = {}
    if dataset_key == 'prostate':
        if dataset_subkey == 'pmri':
            data['train'] = MultisiteMRIProstateDataset(datapath=cfg.data.prostate.pmri.data_path,
                                                        vendor=cfg[net].prostate.pmri.training.vendor,
                                                        split='train',
                                                        load_only_present=cfg[net].prostate.pmri.training.load_only_present)
            if cfg.run.validation:
                data['valid'] = MultisiteMRIProstateDataset(datapath=cfg.data.prostate.pmri.data_path,
                                                        vendor=cfg[net].prostate.pmri.training.vendor,
                                                        split='valid',
                                                        load_only_present=cfg[net].prostate.pmri.training.load_only_present)

    assert len(data) > 0, "No data found to be load"
    return data
