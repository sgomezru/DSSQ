import random
import numpy as np
import torch
from sklearn.metrics import auc
import wandb

### Evaluation
class Metrics(object):
    """
    Object to collect, post-process and log metrics w.r.t. thresholded uncertainty maps.
    
    From binary uncertainty maps and ground truth segmentation, this object calculates
    an adaption of accuracy, precision and recall for uncertainty evaluation. This is used
    mostly to plot Precision-Recall curves, which are the main tool to assess uncertainty
    maps from different origins. All metrics and the plots can then be logged to WandB.
    """

    def __init__(
        self, 
        n_taus
    ):
        self.taus = torch.linspace(0, 1, n_taus).numpy()
        self.reset()
        
    @torch.no_grad()
    def reset(self) -> None:
        self.mse  = 0.
        self.acc  = 0.
        self.rec  = 0.
        self.pre  = 0.
        self.error_rate = 0.
        
        self.tp = 0.
        self.tpfp = 0.
        self.tpfn = 0.
        
    @torch.no_grad()    
    def scale(self, factor: float) -> None:
        self.mse /= factor
        self.acc /= factor
        #self.rec /= factor
        #self.pre /= factor
        self.error_rate /= factor
        
        self.pre = self.tp / self.tpfp
        self.rec = self.tp / self.tpfn
    
    @torch.no_grad()
    def update(self, binary_umaps: Tensor, errmap: Tensor, output=None) -> None:
        if output is not None:
            self.mse += ((output[:1] - output[1:])**2).mean()
            
        self.acc += self._get_accuracy(binary_umaps, errmap)
        #self.rec += self._get_recall(binary_umaps, errmap)
        #self.pre += self._get_precision(binary_umaps, errmap)
        self.error_rate += errmap.sum() / errmap.size(-1)**2

        tp, tpfp, tpfn = self._get_stats(binary_umaps, errmap)
        self.tp += tp
        self.tpfp += tpfp
        self.tpfn += tpfn
        
    @torch.no_grad()   
    def summary_stats(self) -> None:
        self.auc_acc = self.acc.mean()
        self.auc_rec = self.rec.mean()
        self.auc_pre = self.pre.mean()
        self.auc_pr  = auc(self.rec, self.pre) #torch.abs((self.pre[1:] * torch.diff(self.rec, 1))).sum()
        
    @torch.no_grad()    
    def _get_accuracy(self, binary_umaps: Tensor, errmap: Tensor) -> Tensor:
    
        with torch.no_grad():
            t = (binary_umaps == errmap).sum(dim=(0, 2, 3))
            return t / binary_umaps.size(-1)**2
        
    @torch.no_grad()    
    def _get_recall(self, binary_umaps: Tensor, errmap: Tensor) -> Tensor:
    
        with torch.no_grad():
            tp = (binary_umaps * errmap).sum(dim=(0, 2, 3))
            return tp / (errmap == 1).sum().clamp(1)
        
    @torch.no_grad()
    def _get_precision(self, binary_umaps: Tensor, errmap: Tensor) -> Tensor:
    
        with torch.no_grad():
            tp = (binary_umaps * errmap).sum(dim=(0, 2, 3))
            return tp / (binary_umaps == 1).sum(dim=(0, 2, 3)).clamp(1)
        
    @torch.no_grad()    
    def _get_stats(self, binary_umaps: Tensor, errmap: Tensor) -> Tensor:
        
        with torch.no_grad():
            tp   = (binary_umaps * errmap).sum(dim=(0, 2, 3))
            tpfp = (binary_umaps == 1).sum(dim=(0, 2, 3)).clamp(1)
            tpfn = (errmap == 1).sum().clamp(1)
            
            return tp, tpfp, tpfn
    
    @torch.no_grad()
    def log(self):
        data  = [[x, y] for (x, y) in zip(self.rec, self.pre)]
        table = wandb.Table(data=data, columns = ["recall", "precision"])
        xs    = self.taus
        ys    = [self.acc, self.rec, self.pre]
        keys  = ["accuracy", "recall", "precision"]

        wandb.log({
            "pr_auc" :    wandb.plot.line(table, "recall", "precision", 
                                          title="PR curve"),
            "acc_all":    wandb.plot.line_series(xs=xs, ys=ys, keys=keys,
                                                 title="ROC curves", xname='tau'),
            'AUC ACC':    self.auc_acc,
            'AUC REC':    self.auc_rec, 
            'AUC PRE':    self.auc_pre,
            'AUC PR' :    self.auc_pr,
            'Error Rate': self.error_rate
        })

        
        
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=7):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

            
    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        
        return False

def reject_randomness(manualSeed):
    np.random.seed(manualSeed)
    random.rand.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None

def epoch_average(losses, counts):
    losses_np = np.array(losses)
    counts_np = np.array(counts)
    weighted_losses = losses_np * counts_np
    
    return weighted_losses.sum()/counts_np.sum()

def average_metrics(metric_list, counts):
    metrics = dict(metric_list[0])
    counts_np = np.array(counts)
    
    metrics[list(metrics.keys())[0]] = 0
    for layer in list(metrics.keys())[1:]:
        metrics[layer] = dict.fromkeys(metrics[layer], 0)
        
        for metric, count in zip(metric_list, counts_np):
            
            metrics[list(metrics.keys())[0]] += metric[list(metrics.keys())[0]] * count
            for m in metric[layer]:
                metrics[layer][m] += metric[layer][m] * count
        
        metrics[list(metrics.keys())[0]] /= counts_np.sum()
        for m in metrics[layer]:
            metrics[layer][m] /= counts_np.sum()
            
    return metrics
