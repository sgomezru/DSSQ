import torch
import torch.nn as nn
import numpy as np
import pickle
from torch import Tensor
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.manifold import TSNE
from copy import deepcopy
from typing import Tuple

class DimReductAdapter(nn.Module):
    def __init__(self, swivel, n_dims, batch_size, mode='IPCA',
                 pre_fit=False, fit_gaussian=False, project='',
                 device='cuda:0', debug=False):
        super().__init__()
        self.swivel = swivel
        self.n_dims = n_dims
        self.bs = batch_size
        self.mode = mode.upper()
        self.device = device
        self.pre_fit = pre_fit
        self.fit_gaussian = fit_gaussian
        self.project = project
        self.debug = debug
        self.module_path = f'/workspace/out/dim_modules/'
        self.gaussian_path = f'/workspace/out/gaussians/'
        self._init()

    def _init(self):

        self.mu = None
        self.inv_cov = None
        self.activations = None
        self.reduced_acts = []
        self.distances = []
        self.module_path += f'{self.project}_{self.mode}_{self.swivel.replace(".", "_")}_{self.n_dims}dim.pkl'
        self.gaussian_path += f'{self.project}_{self.mode}_gaussian_params_{self.swivel.replace(".", "_")}_{self.n_dims}dim.pt'

        if self.pre_fit is False:
            if self.mode == 'IPCA': self.dim_module = IncrementalPCA(n_components=self.n_dims, batch_size=self.bs)
            elif self.mode == 'PCA': self.dim_module = PCA(n_components=self.n_dims)
            # elif self.mode == 'TSNE': self.dim_module = TSNE(n_components=self.n_dims)
            if self.debug: print(f'Instantiated new {self.mode} module')
        elif self.pre_fit is True:
            try:
                with open(self.module_path, 'rb') as f:
                    self.dim_module = pickle.load(f)
                if self.debug: print(f'Loaded {self.mode} from path {self.module_path}')
            except Exception as e:
                print(f'Unable to load {self.mode} module, error: {e}')
            if self.fit_gaussian is False:
                self._load_gaussian()

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
    
    def _fit_dim_red_module(self):
        assert isinstance(self.activations, np.ndarray), 'Activations required to fit dim reduction model'
        if self.mode in ['PCA']:
            self.dim_module.fit(self.activations)

    def _save_dim_red_module(self):
        try:
            with open(self.module_path, 'wb') as f:
                pickle.dump(self.dim_module, f)
        except Exception as e:
            print(f'Failed saving {self.mode} module, error {e}')

    def _save_gaussian(self):
        assert self.mu is not None and self.inv_cov is not None, 'Mean and inverse covariance matrix required'
        gaussian_params = {'mu': self.mu, 'inv_cov': self.inv_cov}
        torch.save(gaussian_params, self.gaussian_path) 
        if self.debug: print(f'Mean and inverse covariance matrix saved at: {self.gaussian_path}')

    def _load_gaussian(self):
        try:
            gaussian_params = torch.load(self.gaussian_path)
            self.mu = gaussian_params['mu'].to(self.device)
            self.inv_cov = gaussian_params['inv_cov'].to(self.device)
            if self.debug: print(f'Mean and inverse covariance matrix loaded')
        except Exception as e:
            print(f'Failed loading gaussian params, error: {e}')

    def _fit_gaussian(self):
        if isinstance(self.reduced_acts, list): self.reduced_acts = np.vstack(self.reduced_acts)
        self.mu = torch.tensor(np.mean(self.reduced_acts, axis=0)).unsqueeze(0).to(self.device)
        self.inv_cov = torch.tensor(np.linalg.inv(np.cov(self.reduced_acts, rowvar=False))).to(self.device)
        if self.debug: print('Mean and inverse covariance matrix computed and set')
        self._save_gaussian()

    def forward(self, x):
        # X must be of shape (n_samples, n_features), thus flattened, and comes as a torch tensor
        x = x.view(x.size(0), -1)
        x_np = x.detach().cpu().numpy()
        if self.pre_fit is False:
            if self.mode == 'IPCA': self.dim_module.partial_fit(x_np)
            elif self.mode in ['PCA']: self.activations = x_np if self.activations is None else np.vstack([self.activations, x_np])
        elif self.pre_fit is True:
            x_np = self.dim_reduce(x_np)
            if self.fit_gaussian is True: self.reduced_acts.append(x_np)
            else: self._mahalanobis_dist(x_np)

    def dim_reduce(self, x):
        # TSNE doesn't have .transform, instead fit_transform
        return self.dim_module.transform(x)

class DimReductModuleWrapper(nn.Module):
    def __init__(self, model, adapters, copy=True):
        super().__init__()
        self.model = deepcopy(model) if copy else model
        self.adapters = adapters
        self._hook_adapters()
        self.model.eval()

    def _hook_adapters(self):
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

    def set_pre_fit(self):
        for adapter in self.adapters:
            adapter.pre_fit = True
        
    def set_fit_gaussian(self):
        for adapter in self.adapters:
            adapter.fit_gaussian = True

    def save_adapters_modules(self):
        for adapter in self.adapters:
            adapter._save_dim_red_module()

    def fit_adapters_modules(self):
        for adapter in self.adapters:
            adapter._fit_dim_red_module()

    def fit_adapters_gaussians(self):
        for adapter in self.adapters:
            adapter._fit_gaussian()

    def forward(self, x):
        return self.model(x)
