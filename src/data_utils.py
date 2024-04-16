import numpy as np
import torch
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureTyped,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandZoomd,
    SpatialPadd,
    ToTensord,
)

class Transform(object):

    def __init__(self, cfg):
        data_key = cfg['data_key']
        seg_key = cfg['seg_key']
        monai_io_transforms = [
            ToTensord(keys=[data_key, seg_key]),
        ]
        monai_spatial_transforms = [
            RandFlipd(keys=[data_key, seg_key], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=[data_key, seg_key], spatial_axis=[1], prob=0.5)
        ]
        monai_type_transforms = [
            CastToTyped(keys=[data_key, seg_key], dtype=(np.float32, torch.long)),
            EnsureTyped(keys=[data_key, seg_key])
        ]
        self.transforms = {
            'base_transforms': monai_io_transforms + monai_type_transforms,
            'spatial_transforms': monai_spatial_transforms,
            'all_transforms': monai_io_transforms + monai_spatial_transforms + monai_type_transforms
        }

    def get_transforms(self, arg: str):
        assert arg in self.transforms.keys(), f'Argument must be from {self.transforms.keys()}'
        return Compose(self.transforms[arg])

    def __getitem__(self, arg: str):
        return self.get_transforms(arg)
