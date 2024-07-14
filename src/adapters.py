import torch
import torch.nn as nn
import numpy as np
import pickle
from torch import Tensor
from sklearn.decomposition import IncrementalPCA, PCA
from copy import deepcopy
from typing import Tuple

class PCA_Adapter(nn.Module):
    def __init__(self, swivel, n_dims, batch_size, pre_fit=False,
                 train_gaussian=False, compute_dist=False,
                 reduce_dims=True, name='', device='cuda:0',
                 debug=False):
        super().__init__()
        self.swivel = swivel
        self.n_dims = n_dims
        self.bs = batch_size
        self.device = device
        self.pre_fit = pre_fit
        self.reduce_dims = reduce_dims
        self.train_gaussian = train_gaussian
        self.compute_dist = compute_dist
        self.debug = debug
        self.project = name
        self.batch_counter = 0
        self.pca_path = f'/workspace/out/pca/{name}'
        self.act_path = f'/workspace/out/big_acts'
        self._init()

    def _init(self):

        self.mu = None
        self.inv_cov = None
        self.activations = []
        self.distances = []

        if self.pre_fit is False:
            self.pca = IncrementalPCA(n_components=self.n_dims, batch_size=self.bs)
            if self.debug: print('Instantiated new IPCA')
        elif self.pre_fit is True:
            self.pca_path += f'_{self.swivel.replace(".", "_")}'
            self.pca_path += f'_{self.n_dims}dim.pkl'
            try:
                with open(self.pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                if self.debug: print(f'Loaded IPCA from path{self.pca_path}')
            except Exception as e:
                print(f'Unable to load IPCA, error: {e}')
            if self.train_gaussian is False:
                self._load_activations()

    def _load_activations(self):
        try:
            path = f'/workspace/out/activations/{self.project}_{self.swivel.replace(".", "_")}_activations_{self.n_dims}dims.npy'
            self.activations = np.load(path)
            if self.debug: print(f'Loaded activations from path {path}')
            self._set_gaussian()
        except Exception as e:
            print(f'No previously saved activations found {e}')

    @torch.no_grad()
    def _mahalanobis_dist(self, x):
        assert (self.mu is not None and self.inv_cov is not None), "Mean and inverse cov matrix required"
        # x: (n_samples, n_dims)
        # mu: (1, n_dims)
        # inv_cov: (n_dims, n_dims)
        x = torch.tensor(x).to(self.device)
        x_centered = x - self.mu
        mahal_dist = (x_centered @ self.inv_cov * x_centered).sum(dim=1).sqrt()
        self.distances.append(mahal_dist.detach().cpu())

    def _clean_activations(self):
        self.activations = []
        if self.debug: print('Emptied collected activations of adapter')

    def _clean_distances(self):
        self.distances = []
        if self.debug: print('Emptied collected mahalanobis distances of adapter')

    def _set_gaussian(self):
        if isinstance(self.activations, list): self.activations = np.vstack(self.activations)
        self.mu = torch.tensor(np.mean(self.activations, axis=0)).unsqueeze(0).to(self.device)
        self.inv_cov = torch.tensor(np.linalg.inv(np.cov(self.activations, rowvar=False))).to(self.device)
        if self.debug: print('Mean and inverse covariance matrix computed and set')
        self._clean_activations()

    def _save_activations_np(self):
        print('Saving activations...')
        save_path = f'/workspace/out/activations/{self.project}_{self.swivel.replace(".", "_")}_activations_{self.n_dims}dims.npy'
        np.save(save_path, np.vstack(self.activations))

    def forward(self, x):
        # X must be of shape (n_samples, n_features), thus flattened, and comes as a torch tensor
        x = x.view(x.size(0), -1)
        x_np = x.detach().cpu().numpy()
        if self.pre_fit is False:
            if self.swivel == 'model.1.submodule.1.submodule.1.submodule.0.conv' and self.n_dims == 2:
                np.save(f'{self.act_path}/batch_{self.batch_counter}.npy', x_np)
                self.batch_counter += 1
            self.pca.partial_fit(x_np)
        elif self.pre_fit is True:
            if self.reduce_dims is True: x_np = self.dim_reduce(x_np)
            if self.train_gaussian is True: self.activations.append(x_np)
            if self.compute_dist is True: self._mahalanobis_dist(x_np)

    def dim_reduce(self, x):
        return self.pca.transform(x)

class PCAModuleWrapper(nn.Module):
    def __init__(self, model, adapters, copy=True):
        super().__init__()
        self.model = deepcopy(model) if copy else model
        self.adapters = adapters
        self.hook_adapters()
        self.model.eval()

    def hook_adapters(self):
        self.adapter_handles = {}
        for adapter in self.adapters:
            swivel = adapter.swivel
            layer  = self.model.get_submodule(swivel)
            hook   = self._get_hook(adapter)
            self.adapter_handles[swivel] = layer.register_forward_pre_hook(hook)

    def _get_hook( self, adapter):
        def hook_fn( module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            adapter(x[0])
            # return adapter(x)
        return hook_fn

    def forward(self, x):
        return self.model(x)
