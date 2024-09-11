import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultisiteMRIProstateDataset(Dataset):
    """
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
    """

    _VENDORS_INFO = {
        "RUNMC": ["A", "siemens"],
        "BMC": ["B", "philips"],
        "I2CVB": ["C", "siemens"],
        "UCL": ["D", "siemens"],
        "BIDMC": ["E", "ge"],
        "HK": ["F", "siemens"],
    }
    _DS_CONFIG = {"num_classes": 2, "spatial_dims": 2, "size": (384, 384)}

    def __init__(
        self,
        datapath,
        vendor,
        split="all",
        load_only_present=False,
        format="torch",
        transform=None,
        subset=False,
        seed=42,
    ):
        assert vendor in ["siemens", "ge", "philips"], "Invalid vendor"
        assert split in ["all", "train", "valid", "eval"]
        self.vendor = vendor
        self._datapath = Path(datapath).resolve()
        self._split = split
        self._load_only_present = load_only_present
        self._seed = seed
        self._format = format
        self._subset = subset
        self._transform = transform
        self._load_data()

    def _split_subset(self, split_pct=0.8, larger_split="train"):
        num_cases = list(range(len(self.input)))
        split_idx = int(len(num_cases) * split_pct)
        rng = random.Random(self._seed)
        rng.shuffle(num_cases)
        cases = (
            num_cases[:split_idx]
            if self._split == larger_split
            else num_cases[split_idx:]
        )
        self.input = [self.input[i] for i in cases]
        self.target = [self.target[i] for i in cases]

    def _load_data(self):
        self.input = []
        self.target = []
        vendor_sites = [
            site for site, info in self._VENDORS_INFO.items() if info[-1] == self.vendor
        ]
        for site in vendor_sites:
            site_path = self._datapath / site
            for file in site_path.iterdir():
                # Load the ones that have a segmentation associated file to them
                if "segmentation" in file.name.lower():
                    case = file.name[4:6]
                    seg_name = "Segmentation" if site == "BMC" else "segmentation"
                    case_input_path = site_path / f"Case{case}.nii.gz"
                    case_target_path = site_path / f"Case{case}_{seg_name}.nii.gz"
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
        if self._split in ["train", "valid"]:
            self._split_subset(split_pct=0.8, larger_split="train")
            # Subset of the already split data
            if self._subset is True:
                self._split_subset(split_pct=0.3, larger_split=self._split)

        # Subset not a bool this time, just for the eval
        if self._split == "eval" and isinstance(self._subset, str):
            larger_split = self._split if self._subset == "training" else "_"
            self._split_subset(split_pct=0.8, larger_split=larger_split)

        # Concat around last axis (all single slices)
        self.input = np.concatenate(self.input, axis=-1)
        self.input = np.expand_dims(
            self.input, axis=0
        )  # Input final shape after unsqueeze: (1, H, W, Num_slices)
        self.target = np.concatenate(self.target, axis=-1)
        self.target = np.expand_dims(
            self.target, axis=0
        )  # Target final shape after unsqueeze: (1, H, W, Num_slices)
        if self._format == "torch":
            self.input = torch.from_numpy(self.input)
            self.target = torch.from_numpy(self.target)
        # Relabel cases if there are two prostate classes (Since not all datasets distinguish between the two)
        self.target[self.target == 2] = 1

    def __len__(self):
        # Number of slices
        return self.input.shape[-1]

    def __getitem__(self, idx):
        obj = {"input": self.input[..., idx], "target": self.target[..., idx]}
        if self._transform is not None:
            if isinstance(idx, int):
                obj["input"] = np.expand_dims(obj["input"], axis=-1)
                obj["target"] = np.expand_dims(obj["target"], axis=-1)
            obj["input"] = np.transpose(obj["input"], (3, 0, 1, 2))
            obj["target"] = np.transpose(obj["target"], (3, 0, 1, 2))
            obj = self._transform(**obj)
            if isinstance(idx, int):
                obj["input"] = np.squeeze(obj["input"], axis=0)
                obj["target"] = np.squeeze(obj["target"], axis=0)
        return obj


class MnMDataset(Dataset):
    """ """

    def __init__(
        self,
        datapath,
        vendor,
        split="all",
        load_only_present=False,
        format="torch",
        transform=None,
        subset=False,
        one_class=False,
        mode="vendor",
        train_scanner=False,
        n_slices_per_case=3,
        seed=42,
    ):
        assert vendor in ["siemens", "ge", "philips"], "Invalid vendor"
        assert split in ["all", "train", "valid", "eval"]
        assert mode in ["vendor", "scanner"]
        self.vendor = vendor
        self._datapath = Path(datapath).resolve()
        self._split = split
        self._load_only_present = load_only_present
        self._seed = seed
        self._format = format
        self._subset = subset
        self._one_class = one_class
        self._transform = transform
        self._n_slices_per_case = n_slices_per_case
        self._mode = mode
        if self._mode == "scanner":
            self.scanner = "SymphonyTim"
            self._train_scanner = train_scanner
        self._data_info = pd.read_csv(
            self._datapath / "dataset_information.csv", index_col=0
        )
        self._load_data()

    def _split_subset(self, split_pct=0.8, larger_split="train"):
        num_cases = list(range(len(self.input)))
        split_idx = int(len(num_cases) * split_pct)
        rng = random.Random(self._seed)
        rng.shuffle(num_cases)
        cases = (
            num_cases[:split_idx]
            if self._split == larger_split
            else num_cases[split_idx:]
        )
        self.input = [self.input[i] for i in cases]
        self.target = [self.target[i] for i in cases]

    def _crop_center(self, img, height, width):
        h, w = img.shape[:2]
        start_x = max(0, (w - width) // 2)
        start_y = max(0, (h - height) // 2)
        return img[start_y : start_y + height, start_x : start_x + width]

    def _pad_center(self, img, height, width):
        h, w = img.shape[:2]

        # Calculate padding for height
        pad_height_total = max(0, height - h)
        pad_top = pad_height_total // 2
        pad_bottom = pad_height_total - pad_top

        # Calculate padding for width
        pad_width_total = max(0, width - w)
        pad_left = pad_width_total // 2
        pad_right = pad_width_total - pad_left

        # Apply padding
        padded_image = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return padded_image

    def _middle_slices(self, img, n_slices=None):
        img_n_slices = img.shape[-1]
        if n_slices > img_n_slices:
            raise ValueError(
                f"Number of slices per case ({n_slices}) is greater than the number of slices in the case ({img_n_slices})"
            )
        mid_idx = img_n_slices // 2
        start_idx = max(0, mid_idx - n_slices // 2)
        if start_idx + n_slices > img_n_slices:
            start_idx = img_n_slices - n_slices
        return img[..., start_idx : start_idx + n_slices]

    def _pre_process_img(self, img, height=256, width=256, n_slices=None):
        if n_slices is None:
            n_slices = self._n_slices_per_case
        h, w = img.shape[:2]
        if h > height or w > width:
            img = self._crop_center(img, height, width)
        if h < height or w < width:
            img = self._pad_center(img, height, width)
        return self._middle_slices(img, n_slices)

    def _load_data(self):
        self.input = []
        self.target = []
        self.meta = []
        for case in self._data_info.index:
            if (
                self._mode == "vendor"
                and self.vendor in self._data_info.loc[case].VENDOR.lower()
            ) or (
                self._mode == "scanner"
                and (
                    (
                        self._train_scanner
                        and self._data_info.loc[case].SCANNER == self.scanner
                    )
                    or (
                        not self._train_scanner
                        and self._data_info.loc[case].SCANNER != self.scanner
                    )
                )
            ):
                case_path = self._datapath / "dataset" / f"{case:03d}"
                modes = ["ES", "ED"]
                case_img = []
                case_gt = []
                for mode in modes:
                    x = nib.load(case_path / f"{case:03d}_SA_{mode}.nii.gz")
                    y = nib.load(case_path / f"{case:03d}_SA_{mode}_gt.nii.gz")
                    x = self._pre_process_img(x.get_fdata())
                    y = self._pre_process_img(y.get_fdata().astype(int))
                    if self._load_only_present:
                        case_valid_indices = []
                        for slice_idx in range(y.shape[-1]):
                            if len(np.unique(y[..., slice_idx])) > 1:
                                case_valid_indices.append(slice_idx)
                        if not case_valid_indices:
                            continue
                        x = x[..., case_valid_indices]
                        y = y[..., case_valid_indices]
                    case_img.append(x)
                    case_gt.append(y)
                self.input.append(np.concatenate(case_img, axis=-1))
                self.target.append(np.concatenate(case_gt, axis=-1))
                if self._mode == "scanner":
                    self.meta.append(
                        [self._data_info.loc[case].SCANNER] * self.input[-1].shape[-1]
                    )
        # Split here
        if self._split in ["train", "valid"]:
            self._split_subset(split_pct=0.8, larger_split="train")
            # Subset of the already split data
            if self._subset is True:
                self._split_subset(split_pct=0.3, larger_split=self._split)

        # Subset not a bool this time, just for the eval
        if self._split == "eval" and isinstance(self._subset, str):
            larger_split = self._split if self._subset == "training" else "_"
            self._split_subset(split_pct=0.8, larger_split=larger_split)

        # Concat around last axis (all single slices)
        self.input = np.concatenate(self.input, axis=-1)
        self.input = np.expand_dims(
            self.input, axis=0
        )  # Input final shape after unsqueeze: (1, H, W, Num_slices)
        self.target = np.concatenate(self.target, axis=-1)
        self.target = np.expand_dims(
            self.target, axis=0
        )  # Target final shape after unsqueeze: (1, H, W, Num_slices)
        if self._one_class:
            values = np.unique(self.target)
            if len(values) > 2:
                for val in values[2:]:
                    self.target[self.target == val] = 1
        if self._format == "torch":
            self.input = torch.from_numpy(self.input)
            self.target = torch.from_numpy(self.target)

    def __len__(self):
        # Number of slices
        return self.input.shape[-1]

    def __getitem__(self, idx):
        obj = {"input": self.input[..., idx], "target": self.target[..., idx]}
        if self._transform is not None:
            if isinstance(idx, int):
                obj["input"] = np.expand_dims(obj["input"], axis=-1)
                obj["target"] = np.expand_dims(obj["target"], axis=-1)
            obj["input"] = np.transpose(obj["input"], (3, 0, 1, 2))
            obj["target"] = np.transpose(obj["target"], (3, 0, 1, 2))
            obj = self._transform(**obj)
            if isinstance(idx, int):
                obj["input"] = np.squeeze(obj["input"], axis=0)
                obj["target"] = np.squeeze(obj["target"], axis=0)
        return obj
