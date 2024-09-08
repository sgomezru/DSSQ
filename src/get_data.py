import json
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.metrics import DiceMetric
from omegaconf import OmegaConf
from scipy import stats
from torch.utils.data import DataLoader
from sklearn import metrics

REPO_PATH = "/workspace/repositories/DSSQ/src"
OUT_PATH = "/workspace/out"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append(REPO_PATH)

from adapters import DimReductAdapter, DimReductModuleWrapper
from data_utils import get_eval_data
from models import get_unet
from utils import epoch_average

# Set mode
eps = 1e-10
MODE = "eval"
LOG = False
DATA_KEY = "heart"
LOAD_ONLY_PRESENT = True
VALIDATION = True
EXTRA_DESCRIPTION = "_base"
N_DIMS_0 = [2, 4, 8, 16, 32, 64, 128, 256, 512, 960, 1e4]
N_DIMS_PCA = [2, 4, 8, 16, 32, 64, 128, 256, 512, 960]
N_DIMS_IPCA = [2, 4, 8, 16]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

models = ["monai-64-4-4", "swinunetr"]
train_vendors = ["siemens"]
test_vendors = ["siemens", "philips", "ge"]
dim_red_modes = ["IPCA", "PCA", "AVG_POOL"]


def load_conf(unet_name="", iteration=0, data_key="prostate"):
    cfg = OmegaConf.load(f"{REPO_PATH}/configs/conf.yaml")
    OmegaConf.update(cfg, "run.iteration", iteration)
    OmegaConf.update(cfg, "run.data_key", data_key)
    args = unet_name.split("-")
    cfg.wandb.log = LOG
    cfg.unet[data_key].pre = unet_name
    cfg.unet[data_key].arch = args[0]
    cfg.unet[data_key].n_filters_init = int(args[1]) if "monai" in unet_name else None
    cfg.unet[data_key].training.load_only_present = LOAD_ONLY_PRESENT
    cfg.unet[data_key].training.validation = VALIDATION
    cfg.unet[data_key].training.batch_size = 32
    cfg.wandb.project = f"{data_key}_{unet_name}_{iteration}{EXTRA_DESCRIPTION}"
    if MODE == "eval":
        cfg.format = "numpy"
    layer_names = None
    if "monai" in unet_name:
        cfg.unet[data_key].depth = int(args[2])
        cfg.unet[data_key].num_res_units = int(args[3])
        layer_names = [
            f'model.{"1.submodule." * i}0.conv' for i in range(cfg.unet[DATA_KEY].depth)
        ]
        layer_names.append("model.1.submodule.1.submodule.2.0.conv")
    elif unet_name == "swinunetr":
        layer_names = ["encoder2", "encoder3", "encoder4", "encoder10"]
    return cfg, layer_names


def plot_batch(dataset, model, num_images=9, title=None):
    assert 0 < num_images <= 9
    idx = random.sample(range(len(dataset)), num_images)
    data = dataset[idx]
    with torch.no_grad():
        model.eval()
        pred = model(data["input"].cuda()).detach().cpu()
        pred = torch.argmax(pred, dim=1)

    fig, axes = plt.subplots(num_images, 4, figsize=(6, num_images * 2))

    # Iterate over the images and plot them in the grid
    for i in range(num_images):
        if i == 0:
            axes[i][0].set_title("Image")
            axes[i][1].set_title("Target")
            axes[i][2].set_title("Predicted")
            axes[i][3].set_title("Diff")
        axes[i, 0].imshow(data["input"][i][0, ...], cmap="gray")
        axes[i, 1].imshow(data["target"][i][0, ...], cmap="gray")
        axes[i, 2].imshow(pred[i], cmap="gray")
        diff = torch.abs(pred[i] - data["target"][i][0, ...])
        axes[i, 3].imshow(diff, cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")
        axes[i, 3].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=10)
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_roc(fpr, tpr, area, title=None):
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {area:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title is not None:
        plt.title(f"ROC {title}")
    else:
        plt.title(f"ROC")
    plt.legend(loc="lower right")


def eval_set(cfg, model, dataset):
    if cfg.run.data_key == "prostate":
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.unet.prostate.training.batch_size,
            shuffle=False,
            drop_last=False,
        )
        eval_metrics = {
            "Dice Score": DiceMetric(ignore_empty=True, include_background=False)
        }
        metrics = eval_pmri_set(
            model=model, dataloader=dataloader, eval_metrics=eval_metrics
        )
    else:
        raise ValueError(
            f"Invalid data key. No config for dataset named {cfg.run.data_key}"
        )
    return metrics


@torch.no_grad()
def eval_pmri_set(model, dataloader, eval_metrics):
    model.eval()
    epoch_metrics = {key: [] for key in eval_metrics.keys()}
    batch_sizes = []
    for batch in dataloader:
        input_ = batch["input"]
        target = batch["target"]
        batch_sizes.append(input_.shape[0])
        out = model(input_.cuda()).detach().cpu()
        out = torch.argmax(out, dim=1).unsqueeze(1)
        for key, metric in eval_metrics.items():
            computed_metric = metric(out, target).detach().mean().cpu()
            epoch_metrics[key].append(computed_metric)
    for key, epoch_scores in epoch_metrics.items():
        epoch_metrics[key] = epoch_average(epoch_scores, batch_sizes)
    return epoch_metrics


@torch.no_grad()
def eval_pmri_ood(cfg, model, dataset):
    model.empty_data()
    model.eval()
    dice_scores = []
    dm = DiceMetric(ignore_empty=True, include_background=False)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.unet.prostate.training.batch_size,
        shuffle=False,
        drop_last=False,
    )
    ret = {}
    for batch in dataloader:
        input_ = batch["input"]
        target = batch["target"]
        out = model(input_.cuda())
        for key in out:
            out[key] = out[key].detach().cpu()
        for adapter in model.adapters:
            ret[adapter.swivel] = ret.get(adapter.swivel, []) + [
                out[f"{adapter.swivel}_ood"]
            ]
        seg_mask = torch.argmax(out["seg"], dim=1).unsqueeze(1)
        batch_dices = dm(seg_mask, target).detach().cpu()
        dice_scores.append(batch_dices)

    for adapter in model.adapters:
        ret[f"{adapter.swivel}_mahal_dist"] = adapter.distances

    dice_scores = torch.cat(dice_scores, dim=0)
    ret["dice_scores"] = dice_scores.detach().cpu()
    return ret


@torch.no_grad()
def eval_pmri_dice_entropy(cfg, model, dataset):
    if hasattr(model, "empty_data"):
        model.empty_data()
    model.eval()
    probs = []
    dice_scores = []
    dm = DiceMetric(ignore_empty=True, include_background=False)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.unet.prostate.training.batch_size,
        shuffle=False,
        drop_last=False,
    )
    for batch in dataloader:
        input_ = batch["input"].cuda()
        target = batch["target"]
        out = model(input_)
        probs.append(out)
        seg_mask = torch.argmax(out, dim=1).unsqueeze(1)
        batch_dices = dm(seg_mask.detach().cpu(), target)
        dice_scores.append(batch_dices)

    probs = torch.cat(probs, dim=0)
    probs = F.softmax(probs, dim=1)
    probs = probs + eps
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs * log_probs, dim=1)
    entropy = entropy.mean(dim=(1, 2)).detach().cpu()
    dice_scores = torch.cat(dice_scores, dim=0).detach().cpu()
    return dice_scores, entropy, probs.detach().cpu()


@torch.no_grad()
def eval_pmri_MD(cfg, model, dataset):
    """Mahalanobis distance (MD) evaluation"""
    assert isinstance(
        model, DimReductModuleWrapper
    ), "Model must be a DimReductModuleWrapper"
    model.empty_data()
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.unet.prostate.training.batch_size,
        shuffle=False,
        drop_last=False,
    )
    for batch in dataloader:
        input_ = batch["input"]
        model(input_.cuda())

    md = {
        adapter.swivel: torch.cat(adapter.distances, dim=0).detach().cpu()
        for adapter in model.adapters
    }
    return md


def plot_nonzero_example(model, data, arr):
    indices = torch.nonzero(arr == 0)
    idx = indices[0][0]
    plt.imshow(data["eval"][idx]["input"][0].numpy(), cmap="gray")
    plt.imshow(data["eval"][idx]["target"][0].numpy(), cmap="gray")
    out = model(data["eval"][idx]["input"].unsqueeze(0).cuda()).detach().cpu()
    out = torch.argmax(out, dim=1).squeeze().numpy()
    plt.imshow(out, cmap="gray")


def plot_kde(arr):
    data_np = arr.squeeze().numpy()
    kde = stats.gaussian_kde(data_np)
    x_range = np.linspace(data_np.min(), data_np.max(), 100)
    y_kde = kde(x_range)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_kde, "r-", label="KDE")
    plt.hist(data_np, bins=30, density=True, alpha=0.5, label="Histogram")
    plt.title("Distribution of dice scores siemens eval")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def compute_aurc(y_true, y_prob, positive=False, plot=False):
    if positive is False:
        y_prob = -y_prob
    # Sort by confidence scores (highest first)
    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]
    num_samples = len(y_true_sorted)
    coverages = np.arange(1, num_samples + 1) / num_samples
    risks = []
    # Calculate risk and coverage progressively
    cumulative_risk = 0
    for i in range(len(y_true_sorted)):
        # Update risk (misclassification error)
        if (
            y_true_sorted[i] == 0
        ):  # Incorrect prediction (assuming binary labels 0 or 1)
            cumulative_risk += 1
        risks.append(cumulative_risk / (i + 1))  # Risk is cumulative error rate

    # Compute area under the risk-coverage curve (AURC) using trapezoidal rule
    aurc = np.trapz(risks, coverages)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(coverages, risks, label=f"AURC = {aurc:.4f}", color="blue", marker="o")
        plt.fill_between(coverages, risks, color="lightblue", alpha=0.4)
        plt.xlabel("Coverage")
        plt.ylabel("Risk (Error)")
        plt.title("Risk-Coverage Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return aurc


collected_stats_md = []
collected_stats_dim_ent = []
train_md_distances = {}
roc_data = {}
model_data = {}
AUROC_PLOT = False

for vendor in test_vendors:
    cfg, _ = load_conf(data_key=DATA_KEY)
    cfg.unet[DATA_KEY].training.subset = (
        "validation" if vendor in train_vendors else False
    )
    cfg.unet[DATA_KEY].training.vendor = vendor
    data = get_eval_data(train_set=False, val_set=False, eval_set=True, cfg=cfg)
    for iter in range(3):
        for model in models:
            cfg, layer_names = load_conf(model, iteration=iter, data_key=DATA_KEY)
            model_base, state_dict = get_unet(cfg, return_state_dict=True)
            model_base = model_base.to(device)
            model_base.eval()
            model_base_dices, model_base_entropy, model_base_probs = (
                eval_pmri_dice_entropy(cfg, model_base, data["eval"])
            )
            model_base_probs = model_base_probs.to(device)
            model_data[f"{model}_{vendor}_test_{iter}"] = {
                "mean_dice": model_base_dices.mean().item(),
                "std_dice": model_base_dices.std().item(),
                "mean_entropy": model_base_entropy.mean().item(),
                "std_entropy": model_base_entropy.std().item(),
            }
            if vendor == "siemens":
                ood_dice_th_5 = torch.quantile(model_base_dices, 0.05).item()
                ood_dice_th_10 = torch.quantile(model_base_dices, 0.10).item()
                model_data[f"{model}_{vendor}_test_{iter}"]["ood_dice_th_5"] = (
                    ood_dice_th_5
                )
                model_data[f"{model}_{vendor}_test_{iter}"]["ood_dice_th_10"] = (
                    ood_dice_th_10
                )
            else:
                ood_dice_th_10 = model_data[f"{model}_siemens_test_{iter}"][
                    "ood_dice_th_10"
                ]
                ood_dice_th_5 = model_data[f"{model}_siemens_test_{iter}"][
                    "ood_dice_th_5"
                ]
            ood_mask_5 = (model_base_dices < ood_dice_th_5).squeeze()
            ood_mask_10 = (model_base_dices < ood_dice_th_10).squeeze()
            ood_mask_95 = (model_base_dices < 0.95).squeeze()
            for dim_red_mode in dim_red_modes:
                if dim_red_mode == "AVG_POOL":
                    n_dims_iter = [1e4]
                elif dim_red_mode == "PCA":
                    n_dims_iter = [2, 4, 8, 16, 512, 960]
                elif dim_red_mode == "IPCA":
                    n_dims_iter = [2, 4, 8, 16]
                for n_dims in n_dims_iter:
                    print(
                        f"Running iter {iter} for vendor {vendor}, {dim_red_mode} mode of {n_dims} dims"
                    )
                    adapters = [
                        DimReductAdapter(
                            swivel,
                            n_dims,
                            cfg.unet[DATA_KEY].training.batch_size,
                            mode=dim_red_mode,
                            pre_fit=True,
                            fit_gaussian=False,
                            project=cfg.wandb.project,
                        )
                        for swivel in layer_names
                    ]
                    adapters = nn.ModuleList(adapters)
                    model_adapted = DimReductModuleWrapper(
                        model=model_base, adapters=adapters
                    )
                    model_adapted = model_adapted.to(device)
                    model_adapted.eval()
                    # Mahalanobis distances (md)
                    print("MD eval")
                    md = eval_pmri_MD(cfg, model_adapted, data["eval"])
                    for adapter in md:
                        tmp_data = md[adapter]
                        if vendor == "siemens":
                            reg_min = tmp_data.min().item()
                            reg_max = tmp_data.max().item()
                            train_md_distances[
                                f"{model}_{adapter}_{iter}_{dim_red_mode}_{n_dims}_val"
                            ] = {"min": reg_min, "max": reg_max}
                        else:
                            reg_min = np.min(
                                [
                                    train_md_distances[
                                        f"{model}_{adapter}_{iter}_{dim_red_mode}_{n_dims}_val"
                                    ]["min"],
                                    tmp_data.min().item(),
                                ]
                            )
                            reg_max = np.max(
                                [
                                    train_md_distances[
                                        f"{model}_{adapter}_{iter}_{dim_red_mode}_{n_dims}_val"
                                    ]["max"],
                                    tmp_data.max().item(),
                                ]
                            )
                        # norm_tmp_data = (tmp_data - tmp_data.min()) / (tmp_data.max() - tmp_data.min())
                        norm_tmp_data = (tmp_data - reg_min) / (reg_max - reg_min)
                        ood_data_5 = tmp_data[ood_mask_5]
                        id_data_5 = tmp_data[~ood_mask_5]
                        aurc5 = compute_aurc(
                            ood_mask_5.numpy(), norm_tmp_data.numpy(), positive=False
                        )
                        fpr5, tpr5, ths5 = metrics.roc_curve(
                            ood_mask_5.long().numpy(), norm_tmp_data.numpy()
                        )
                        ood_data_10 = tmp_data[ood_mask_10]
                        id_data_10 = tmp_data[~ood_mask_10]
                        aurc10 = compute_aurc(
                            ood_mask_10.numpy(), norm_tmp_data.numpy(), positive=False
                        )
                        fpr10, tpr10, ths10 = metrics.roc_curve(
                            ood_mask_10.long().numpy(), norm_tmp_data.numpy()
                        )
                        ood_data_95 = tmp_data[ood_mask_95]
                        id_data_95 = tmp_data[~ood_mask_95]
                        aurc95 = compute_aurc(
                            ood_mask_95.numpy(), norm_tmp_data.numpy(), positive=False
                        )
                        fpr95, tpr95, ths95 = metrics.roc_curve(
                            ood_mask_95.long().numpy(), norm_tmp_data.numpy()
                        )
                        collected_stats_md.append(
                            {
                                "model": model,
                                "dim_red_mode": dim_red_mode,
                                "n_dims": n_dims,
                                "vendor": f"{vendor}_test",
                                "iteration": iter,
                                "layer": adapter,
                                "md_mean": tmp_data.mean().item(),
                                "md_std": tmp_data.mean().item(),
                                "sprcorr": stats.spearmanr(
                                    model_base_dices.numpy(), tmp_data.numpy()
                                )[0],
                                "num_ood_5": ood_data_5.shape[0],
                                "num_id_5": id_data_5.shape[0],
                                "md_ood_5_mean": tmp_data[ood_mask_5].mean().item(),
                                "md_ood_5_std": tmp_data[ood_mask_5].std().item(),
                                "md_id_5_mean": tmp_data[~ood_mask_5].mean().item(),
                                "md_id_5_std": tmp_data[~ood_mask_5].std().item(),
                                "aurc_5": aurc5,
                                "auroc_5": metrics.auc(fpr5, tpr5),
                                "auprc_5": metrics.average_precision_score(
                                    ood_mask_5.long().numpy(), norm_tmp_data.numpy()
                                ),
                                "num_ood_10": ood_data_10.shape[0],
                                "num_id_10": id_data_10.shape[0],
                                "md_ood_10_mean": tmp_data[ood_mask_10].mean().item(),
                                "md_ood_10_std": tmp_data[ood_mask_10].std().item(),
                                "md_id_10_mean": tmp_data[~ood_mask_10].mean().item(),
                                "md_id_10_std": tmp_data[~ood_mask_10].std().item(),
                                "aurc_10": aurc10,
                                "auroc_10": metrics.auc(fpr10, tpr10),
                                "auprc_10": metrics.average_precision_score(
                                    ood_mask_10.long().numpy(), norm_tmp_data.numpy()
                                ),
                                "num_ood_95": ood_data_95.shape[0],
                                "num_id_95": id_data_95.shape[0],
                                "md_ood_95_mean": tmp_data[ood_mask_95].mean().item(),
                                "md_ood_95_std": tmp_data[ood_mask_95].std().item(),
                                "md_id_95_mean": tmp_data[~ood_mask_95].mean().item(),
                                "md_id_95_std": tmp_data[~ood_mask_95].std().item(),
                                "aurc_95": aurc95,
                                "auroc_95": metrics.auc(fpr95, tpr95),
                                "auprc_95": metrics.average_precision_score(
                                    ood_mask_95.long().numpy(), norm_tmp_data.numpy()
                                ),
                            }
                        )
                        # if AUROC_PLOT:
                        #     sorted_mahal_indices = np.argsort(tmp_data.numpy())
                        #     sorted_dice_scores = model_base_dices.numpy()[sorted_mahal_indices]
                        #     sorted_iid = ~ood_mask_5.numpy()[sorted_mahal_indices]
                        #     error_rates = []
                        #     fractions_rejected = []
                        #     for i in range(len(sorted_dice_scores)):
                        #         fraction_rejected = i / len(sorted_dice_scores)
                        #         remaining_id = sorted_iid[i:]
                        #         error_rate = np.sum(~remaining_id) / ood_data_5.shape[0]
                        #         fractions_rejected.append(fraction_rejected)
                        #         error_rates.append(error_rate)
                        #     plt.figure(figsize=(8,6))
                        #     plt.plot(fractions_rejected, error_rates, marker='o')
                        #     plt.xlabel('Fraction images rejected')
                        #     plt.ylabel('Error rate (Fraction of OOD images remaining)')
                        #     plt.title(f'Error rate vs fraction images rejected for {model}_{adapter}_{iter}_{dim_red_mode}_{n_dims}_{vendor}')
                        #     plt.grid(True)
                        #     plt.tight_layout()
                        #     plt.savefig(f'{OUT_PATH}/eval_images/{model}_{adapter}_{iter}_{dim_red_mode}_{n_dims}_{vendor}_erc.png')
                        #     plt.close()
                    for layer in layer_names:
                        adapters = [
                            DimReductAdapter(
                                layer,
                                n_dims,
                                cfg.unet[DATA_KEY].training.batch_size,
                                mode=dim_red_mode,
                                pre_fit=True,
                                fit_gaussian=False,
                                undo_dim=True,
                                project=cfg.wandb.project,
                            )
                        ]
                        adapters = nn.ModuleList(adapters)
                        model_adapted = DimReductModuleWrapper(
                            model=model_base, adapters=adapters, upstream_hooks=True
                        )
                        model_adapted = model_adapted.to(device)
                        model_adapted.eval()
                        print(f"Anomaly score diff {layer}")
                        mod_dices, mod_entropy, mod_probs = eval_pmri_dice_entropy(
                            cfg, model_adapted, data["eval"]
                        )
                        mod_probs = mod_probs.to(device)
                        diff = torch.abs(model_base_probs - mod_probs)
                        diff = diff.mean(dim=(1, 2, 3)).detach().cpu()
                        fpr5diff, tpr5diff, ths5diff = metrics.roc_curve(
                            ood_mask_5.long().numpy(), diff.numpy()
                        )
                        aurc5diff = compute_aurc(
                            ood_mask_5.numpy(), diff.numpy(), positive=False
                        )
                        fpr10diff, tpr10diff, ths10diff = metrics.roc_curve(
                            ood_mask_10.long().numpy(), diff.numpy()
                        )
                        aurc10diff = compute_aurc(
                            ood_mask_10.numpy(), diff.numpy(), positive=False
                        )
                        fpr95diff, tpr95diff, ths95diff = metrics.roc_curve(
                            ood_mask_95.long().numpy(), diff.numpy()
                        )
                        aurc95diff = compute_aurc(
                            ood_mask_95.numpy(), diff.numpy(), positive=False
                        )
                        fpr5ent, tpr5ent, ths5ent = metrics.roc_curve(
                            ood_mask_5.long().numpy(), mod_entropy.numpy()
                        )
                        aurc5ent = compute_aurc(
                            ood_mask_5.numpy(), mod_entropy.numpy(), positive=True
                        )
                        fpr10ent, tpr10ent, ths10ent = metrics.roc_curve(
                            ood_mask_10.long().numpy(), mod_entropy.numpy()
                        )
                        aurc10ent = compute_aurc(
                            ood_mask_10.numpy(), mod_entropy.numpy(), positive=True
                        )
                        fpr95ent, tpr95ent, ths95ent = metrics.roc_curve(
                            ood_mask_95.long().numpy(), mod_entropy.numpy()
                        )
                        aurc95ent = compute_aurc(
                            ood_mask_95.numpy(), mod_entropy.numpy(), positive=True
                        )
                        collected_stats_dim_ent.append(
                            {
                                "model": model,
                                "dim_red_mode": dim_red_mode,
                                "n_dims": n_dims,
                                "vendor": f"{vendor}_test",
                                "iteration": iter,
                                "layer": layer,
                                "entropy_mean": mod_entropy.mean().item(),
                                "entropy_std": mod_entropy.std().item(),
                                "entropy_ood_5_mean": mod_entropy[ood_mask_5]
                                .mean()
                                .item(),
                                "entropy_ood_5_std": mod_entropy[ood_mask_5]
                                .std()
                                .item(),
                                "entropy_id_5_mean": mod_entropy[~ood_mask_5]
                                .mean()
                                .item(),
                                "entropy_id_5_std": mod_entropy[~ood_mask_5]
                                .std()
                                .item(),
                                "entropy_ood_10_mean": mod_entropy[ood_mask_10]
                                .mean()
                                .item(),
                                "entropy_ood_10_std": mod_entropy[ood_mask_10]
                                .std()
                                .item(),
                                "entropy_id_10_mean": mod_entropy[~ood_mask_10]
                                .mean()
                                .item(),
                                "entropy_id_10_std": mod_entropy[~ood_mask_10]
                                .std()
                                .item(),
                                "entropy_ood_95_mean": mod_entropy[ood_mask_95]
                                .mean()
                                .item(),
                                "entropy_ood_95_std": mod_entropy[ood_mask_95]
                                .std()
                                .item(),
                                "aurc_5_ent": aurc5ent,
                                "auroc_5_ent": metrics.auc(fpr5ent, tpr5ent),
                                "auprc_5_ent": metrics.average_precision_score(
                                    ood_mask_5.long().numpy(), mod_entropy.numpy()
                                ),
                                "aurc_10_ent": aurc10ent,
                                "auroc_10_ent": metrics.auc(fpr10ent, tpr10ent),
                                "auprc_10_ent": metrics.average_precision_score(
                                    ood_mask_10.long().numpy(), mod_entropy.numpy()
                                ),
                                "aurc_95_ent": aurc95ent,
                                "auroc_95_ent": metrics.auc(fpr95ent, tpr95ent),
                                "auprc_95_ent": metrics.average_precision_score(
                                    ood_mask_95.long().numpy(), mod_entropy.numpy()
                                ),
                                "anomaly_diff_mean": diff.mean().item(),
                                "anomaly_diff_std": diff.std().item(),
                                "anomaly_diff_ood_5_mean": diff[ood_mask_5]
                                .mean()
                                .item(),
                                "anomaly_diff_ood_5_std": diff[ood_mask_5].std().item(),
                                "anomaly_diff_id_5_mean": diff[~ood_mask_5]
                                .mean()
                                .item(),
                                "anomaly_diff_id_5_std": diff[~ood_mask_5].std().item(),
                                "anomaly_diff_ood_10_mean": diff[ood_mask_10]
                                .mean()
                                .item(),
                                "anomaly_diff_ood_10_std": diff[ood_mask_10]
                                .std()
                                .item(),
                                "anomaly_diff_id_10_mean": diff[~ood_mask_10]
                                .mean()
                                .item(),
                                "anomaly_diff_id_10_std": diff[~ood_mask_10]
                                .std()
                                .item(),
                                "anomaly_diff_ood_95_mean": diff[ood_mask_95]
                                .mean()
                                .item(),
                                "anomaly_diff_ood_95_std": diff[ood_mask_95]
                                .std()
                                .item(),
                                "aurc_5_diff": aurc5diff,
                                "auroc_5_diff": metrics.auc(fpr5diff, tpr5diff),
                                "auprc_5_diff": metrics.average_precision_score(
                                    ood_mask_5.long().numpy(), diff.numpy()
                                ),
                                "aurc_10": aurc10diff,
                                "auroc_10_diff": metrics.auc(fpr10diff, tpr10diff),
                                "auprc_10_diff": metrics.average_precision_score(
                                    ood_mask_10.long().numpy(), diff.numpy()
                                ),
                                "aurc_95_diff": aurc95diff,
                                "auroc_95_diff": metrics.auc(fpr95diff, tpr95diff),
                                "auprc_95_diff": metrics.average_precision_score(
                                    ood_mask_95.long().numpy(), diff.numpy()
                                ),
                            }
                        )


df_md = pd.DataFrame(collected_stats_md)
df_md.to_csv(f"{OUT_PATH}/eval_data/mahal_dist_stats_heart.csv", index=False)
df_dim_ent = pd.DataFrame(collected_stats_dim_ent)
df_dim_ent.to_csv(f"{OUT_PATH}/eval_data/dim_ent_stats_heart.csv", index=False)
with open(f"{OUT_PATH}/eval_data/model_stats_heart.json", "w") as f:
    json.dump(model_data, f)