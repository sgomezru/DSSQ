import torch
import torch.nn as nn
import numpy as np
import pickle
from torch import Tensor
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from typing import Tuple


class DimReductAdapter(nn.Module):
    def __init__(
        self,
        swivel,
        n_dims,
        batch_size,
        mode="IPCA",
        pre_fit=False,
        fit_gaussian=False,
        fit_scaler=False,
        undo_dim=False,
        store_acts=False,
        project="",
        device="cuda:0",
        debug=False,
    ):
        super().__init__()
        self.swivel = swivel
        self.n_dims = n_dims
        self.bs = batch_size
        self.mode = mode.upper()
        self.device = device
        self.pre_fit = pre_fit
        self.fit_gaussian = fit_gaussian
        self.fit_scaler = fit_scaler
        self.undo_dim = undo_dim
        self.store_acts = store_acts
        self.project = project
        self.debug = debug
        self.module_path = "/workspace/out/dms/"
        self.gaussian_path = "/workspace/out/gms/"
        self._init()

    def _init(self):
        self.mu = None
        self.inv_cov = None
        self._clean_storage()
        self.scaler_path = f'{self.module_path}{self.project}_{self.swivel.replace(".", "_")}_scaler.pkl'
        self.module_path += (
            f'{self.project}_{self.mode}_{self.swivel.replace(".", "_")}.pkl'
        )
        self.gaussian_path += f'{self.project}_{self.mode}_gaussian_{self.swivel.replace(".", "_")}_{self.n_dims}dim.pt'
        if self.fit_scaler is False and "PCA" in self.mode:
            try:
                with open(self.scaler_path, "rb") as f:
                    self.dim_scaler = pickle.load(f)
            except Exception as e:
                print(f"Unable to load scaler, error: {e}")
        elif self.fit_scaler is True and "PCA" in self.mode:
            self.dim_scaler = StandardScaler()
        if self.mode == "AVG_POOL":
            # self.module_path = self.module_path.split("_scaled")[0] + ".pkl"
            # self.gaussian_path = self.gaussian_path.split("_scaled")[0] + ".pt"
            self.dim_module = nn.AvgPool2d(kernel_size=2, stride=2)
            if self.debug:
                print(f"Instantiated new {self.mode} module")
        if self.pre_fit is False and self.fit_scaler is False:
            if self.mode == "IPCA":
                self.dim_module = IncrementalPCA(n_components=32, batch_size=self.bs)
            elif self.mode == "PCA":
                self.dim_module = PCA(n_components=self.n_dims)
            if self.debug:
                print(f"Instantiated new {self.mode} module")
        elif self.pre_fit is True:
            if "PCA" in self.mode:
                try:
                    with open(self.module_path, "rb") as f:
                        tmp_dim_module = pickle.load(f)
                        self.dim_module = PCA(n_components=self.n_dims)
                        self.dim_module.mean_ = tmp_dim_module.mean_
                        self.dim_module.components_ = tmp_dim_module.components_[
                            : self.n_dims
                        ]
                    if self.debug:
                        print(f"Loaded {self.mode} from path {self.module_path}")
                except Exception as e:
                    print(f"Unable to load {self.mode} module, error: {e}")
            if self.fit_gaussian is False:
                self._load_gaussian()

    def _clean_storage(self):
        self.activations = None
        self.reduced_acts = []
        self.distances = []

    @torch.no_grad()
    def _mahalanobis_dist(self, x):
        assert (
            self.mu is not None and self.inv_cov is not None
        ), "Mean and inverse cov matrix required"
        # x: (n_samples, n_dims)
        # mu: (1, n_dims)
        # inv_cov: (n_dims, n_dims)
        x = torch.tensor(x, dtype=torch.float64).to(self.device)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Nan or Inf values found in x input {self.swivel}")
        x_centered = x - self.mu
        mahal_dist = ((x_centered @ self.inv_cov) * x_centered).sum(dim=1).sqrt()
        # mahal_dist = x_centered @ self.inv_cov
        # mahal_dist = mahal_dist * x_centered
        # mahal_dist = mahal_dist.sum(dim=1).sqrt()
        self.distances.append(mahal_dist.detach().cpu())

    def _fit_dim_red_module(self):
        assert isinstance(
            self.activations, np.ndarray
        ), "Activations required to fit dim reduction model"
        if self.mode in ["PCA"]:
            self.dim_module.fit(self.activations)

    def _save_dim_scaler(self):
        try:
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.dim_scaler, f)
        except Exception as e:
            print(f"Failed saving scaler, error {e}")

    def _save_dim_red_module(self):
        try:
            with open(self.module_path, "wb") as f:
                pickle.dump(self.dim_module, f)
        except Exception as e:
            print(f"Failed saving {self.mode} module, error {e}")

    def _save_gaussian(self):
        assert (
            self.mu is not None and self.inv_cov is not None
        ), "Mean and inverse covariance matrix required"
        gaussian_params = {"mu": self.mu, "inv_cov": self.inv_cov}
        torch.save(gaussian_params, self.gaussian_path)
        if self.debug:
            print(f"Mean and inverse covariance matrix saved at: {self.gaussian_path}")

    def _load_gaussian(self):
        try:
            gaussian_params = torch.load(self.gaussian_path)
            self.mu = gaussian_params["mu"].to(self.device)
            self.inv_cov = gaussian_params["inv_cov"].to(self.device)
            if self.debug:
                print("Mean and inverse covariance matrix loaded")
        except Exception as e:
            print(f"Failed loading gaussian params, error: {e}")

    def _check_positive_definite(self, mat):
        try:
            np.linalg.cholesky(mat)
            return True
        except np.linalg.LinAlgError:
            return False

    def _compute_inv_cholesky(self, cov):
        L = np.linalg.cholesky(cov)
        inv_cov = np.linalg.inv(L).T @ np.linalg.inv(L)
        return inv_cov

    def _fit_gaussian(self):
        if isinstance(self.reduced_acts, list):
            self.reduced_acts = np.vstack(self.reduced_acts)
        self.mu = (
            torch.tensor(np.mean(self.reduced_acts, axis=0))
            .unsqueeze(0)
            .to(self.device)
        )
        cov = np.cov(self.reduced_acts, rowvar=False)
        if self._check_positive_definite(cov) and self._check_positive_definite(
            self._compute_inv_cholesky(cov)
        ):
            self.inv_cov = torch.tensor(self._compute_inv_cholesky(cov)).to(self.device)
        else:
            cov = cov + 1e-10 * np.eye(cov.shape[0])
            if self._check_positive_definite(cov) and self._check_positive_definite(
                self._compute_inv_cholesky(cov)
            ):
                self.inv_cov = torch.tensor(self._compute_inv_cholesky(cov)).to(
                    self.device
                )
            else:
                print(f"Failed to compute inverse covariance matrix for {self.swivel}")
        if self.debug:
            print("Mean and inverse covariance matrix computed and set")
        self._save_gaussian()

    def _set_threshold(self, th_val):
        self.th_val = th_val

    def compute_threshold(self, percentile_cut):
        assert len(self.distances) > 0, "Must have previously computed distances"
        tmp_dists = torch.cat(self.distances, dim=0).detach().cpu()
        self.th_val = torch.quantile(tmp_dists, percentile_cut).item()

    def compute_ood(self):
        assert self.th_val is not None, "Threshold value must be initially defined"
        assert len(self.distances) > 0, "Must have previously computed distances"
        arr = self.distances[-1] > self.th_val
        return arr

    def dim_reduce(self, x):
        if self.mode in ["PCA", "IPCA"]:
            dm = self.dim_module.transform(x)
            return dm
        elif self.mode == "AVG_POOL":
            while torch.prod(torch.tensor(x.size()[1:])) > self.n_dims:
                x = self.dim_module(x)
            x = self.dim_module(x)
            return x

    def undo_dim_reduce(self, x, x_shape):
        x_reversed = self.dim_module.inverse_transform(x)
        x_reversed = torch.tensor(x_reversed, dtype=torch.float).to(self.device)
        x_reversed = x_reversed.view(x_shape)
        return x_reversed

    def forward(self, x):
        # X must be of shape (n_samples, n_features), thus flattened, and comes as a torch tensor
        # print(f"Adapter {self.swivel} received input of shape {x.size()}")
        x = x.detach().cpu()
        if "PCA" in self.mode:
            x_shape = x.size()
            x = x.contiguous().view(x.size(0), -1)
            x_np = x.detach().cpu().numpy()
            if self.fit_scaler is True:
                self.dim_scaler.partial_fit(x_np)
            elif self.pre_fit is False:
                x_np = self.dim_scaler.transform(x_np)
                if self.mode == "IPCA":
                    self.dim_module.partial_fit(x_np)
                elif self.mode == "PCA":
                    self.activations = (
                        x_np
                        if self.activations is None
                        else np.vstack([self.activations, x_np])
                    )
            elif self.pre_fit is True:
                x_sc = self.dim_scaler.transform(x_np)
                x_sc = self.dim_reduce(x_sc)
                if self.fit_gaussian is True:
                    self.reduced_acts.append(x_sc)
                elif self.undo_dim is True:
                    x_np = self.dim_reduce(x_np)
                    x_reversed = self.undo_dim_reduce(x_np, x_shape)
                    return x_reversed
                else:
                    if self.store_acts is True:
                        self.reduced_acts.append(x_sc)
                    self._mahalanobis_dist(x_sc)
        elif self.mode == "AVG_POOL":
            x = self.dim_reduce(x)
            x_np = x.contiguous().view(x.size(0), -1).detach().cpu().numpy()
            if self.fit_gaussian is True:
                self.reduced_acts.append(x_np)
            else:
                self._mahalanobis_dist(x_np)


class DimReductModuleWrapper(nn.Module):
    def __init__(
        self, model, adapters, downstream_ood=False, upstream_hooks=False, copy=True
    ):
        super().__init__()
        self.model = deepcopy(model) if copy else model
        self.downstream_ood = downstream_ood
        self.adapters = adapters
        self._upstream_hooks = upstream_hooks
        self._hook_adapters()
        self.model.eval()

    def _hook_adapters(self):
        self.adapter_handles = {}
        for adapter in self.adapters:
            swivel = adapter.swivel
            layer = self.model.get_submodule(swivel)
            hook = self._get_hook(adapter)
            self.adapter_handles[swivel] = layer.register_forward_pre_hook(hook)

    def _get_hook(self, adapter):
        def hook_fn(module: nn.Module, x: Tuple[Tensor]) -> Tensor:
            if self._upstream_hooks is False:
                adapter(x[0])
            else:
                return adapter(x[0])

        return hook_fn

    def set_downstream_ood(self, val):
        assert isinstance(val, bool), "Flag must be boolean"
        self.downstream_ood = val

    def set_pre_fit_flag(self, val):
        assert isinstance(val, bool), "Flag must be boolean"
        for adapter in self.adapters:
            adapter.pre_fit = val

    def set_fit_gaussian_flag(self, val):
        assert isinstance(val, bool), "Flag must be boolean"
        for adapter in self.adapters:
            adapter.fit_gaussian = val

    def save_adapters_modules(self):
        for adapter in self.adapters:
            adapter._save_dim_red_module()

    def save_scalers_modules(self):
        for adapter in self.adapters:
            adapter._save_dim_scaler()

    def save_adapters_thresholds(self):
        for adapter in self.adapters:
            adapter._save_threshold()

    def fit_adapters_modules(self):
        for adapter in self.adapters:
            adapter._fit_dim_red_module()

    def fit_adapters_gaussians(self):
        for adapter in self.adapters:
            adapter._fit_gaussian()

    def set_thresholds(self, thresholds_dict):
        for adapter in self.adapters:
            adapter._set_threshold(thresholds_dict[adapter.swivel])

    def empty_data(self):
        for adapter in self.adapters:
            adapter._clean_storage()

    def compute_thresholds(self, percentile_cut):
        thresholds = {}
        for adapter in self.adapters:
            adapter.compute_threshold(percentile_cut)
            thresholds[adapter.swivel] = adapter.th_val
        return thresholds

    def forward(self, x):
        ret_fw = self.model(x)
        if self.downstream_ood is True:
            ret_fw = {"seg": ret_fw}
            for adapter in self.adapters:
                ret_fw[f"{adapter.swivel}_ood"] = adapter.compute_ood()
        return ret_fw
