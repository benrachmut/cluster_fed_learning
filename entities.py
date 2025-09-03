import copy
from collections import defaultdict
from random import Random

import torchvision
from sympy.abc import epsilon
from sympy.physics.units import amount
from torch.utils.data import DataLoader, Dataset
from config import *
import torch.nn.functional as F
from itertools import combinations
from sklearn.cluster import KMeans
import torchvision.models as models

from abc import ABC, abstractmethod

# Define AlexNet for clients

import numpy as np

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- utilities ----------
class WithIndex(torch.utils.data.Dataset):
    """Wrap a dataset so __getitem__ returns (..., idx)."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        if isinstance(item, tuple):
            return (*item, idx)
        return item, idx

class AlexNet(nn.Module):
    def __init__(self, num_classes, num_clusters=1):
        super(AlexNet, self).__init__()

        # Backbone (shared layers)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout()
        )

        # If there's only 1 head, keep it simple (like original)
        if num_clusters == 1:
            self.head = nn.Linear(4096, num_classes)  # Single head
        else:
            # Multi-head with deeper structures
            self.heads = nn.ModuleDict({
                f"head_{i}": nn.Sequential(
                    nn.Linear(4096, 2048), nn.ReLU(),
                    nn.Linear(2048, 1024), nn.ReLU(),
                    nn.Linear(1024, num_classes)
                ) for i in range(num_clusters)
            })

        self.num_clusters = num_clusters

    def forward(self, x, cluster_id=None):
        x = self.backbone(x)

        if self.num_clusters == 1:
            return self.head(x)  # Single-head case

        if cluster_id is not None:
            return self.heads[f"head_{cluster_id}"](x)  # Select head by cluster_id

        # If no cluster_id is given, return outputs from all heads
        return {f"head_{i}": head(x) for i, head in self.heads.items()}



# Define VGG16 for server
class VGGServer(nn.Module):
    def __init__(self, num_classes, num_clusters=1):
        super(VGGServer, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=None)  # No pre-trained weights
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)  # Adjust for final output

        self.num_clusters = num_clusters

        if num_clusters == 1:
            # Single head case
            self.head = nn.Linear(num_classes, num_classes)
        else:
            # Multi-head with deeper layers
            self.heads = nn.ModuleDict({
                f"head_{i}": nn.Sequential(
                    nn.Linear(num_classes, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, num_classes)
                ) for i in range(num_clusters)
            })

    def forward(self, x, cluster_id=None):
        # Resize input to match VGG's expected input size
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.vgg(x)  # Backbone output

        if self.num_clusters == 1:
            return self.head(x)  # Single-head case

        if cluster_id is not None:
            return self.heads[f"head_{cluster_id}"](x)  # Select head by cluster_id

        # If no cluster_id is given, return outputs from all heads
        return {f"head_{i}": head(x) for i, head in self.heads.items()}

# -----------------------
# ResNet18 (light/residual)
# -----------------------
class ResNet18Server(nn.Module):
    def __init__(self, num_classes, num_clusters=1):
        super().__init__()
        self.net = torchvision.models.resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

        self.num_clusters = num_clusters
        if num_clusters == 1:
            self.head = nn.Linear(num_classes, num_classes)
        else:
            self.heads = nn.ModuleDict({
                f"head_{i}": nn.Sequential(
                    nn.Linear(num_classes, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, num_classes)
                ) for i in range(num_clusters)
            })

    def forward(self, x, cluster_id=None):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.net(x)
        if self.num_clusters == 1:
            return self.head(x)
        if cluster_id is not None:
            return self.heads[f"head_{cluster_id}"](x)
        return {f"head_{i}": head(x) for i, head in self.heads.items()}


# -----------------------
# MobileNetV2 (very lightweight/mobile)
# -----------------------
class MobileNetV2Server(nn.Module):
    def __init__(self, num_classes, num_clusters=1):
        super().__init__()
        self.net = torchvision.models.mobilenet_v2(weights=None)
        # Replace final classifier layer
        in_feats = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_feats, num_classes)

        self.num_clusters = num_clusters
        if num_clusters == 1:
            self.head = nn.Linear(num_classes, num_classes)
        else:
            self.heads = nn.ModuleDict({
                f"head_{i}": nn.Sequential(
                    nn.Linear(num_classes, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, num_classes)
                ) for i in range(num_clusters)
            })

    def forward(self, x, cluster_id=None):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.net(x)
        if self.num_clusters == 1:
            return self.head(x)
        if cluster_id is not None:
            return self.heads[f"head_{cluster_id}"](x)
        return {f"head_{i}": head(x) for i, head in self.heads.items()}


# -----------------------
# SqueezeNet 1.1 (tiny params)
# -----------------------
class SqueezeNetServer(nn.Module):
    def __init__(self, num_classes, num_clusters=1):
        super().__init__()
        self.net = torchvision.models.squeezenet1_1(weights=None)
        # SqueezeNet uses a conv classifier head
        self.net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.net.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # ensure flat logits

        self.num_clusters = num_clusters
        if num_clusters == 1:
            self.head = nn.Linear(num_classes, num_classes)
        else:
            self.heads = nn.ModuleDict({
                f"head_{i}": nn.Sequential(
                    nn.Linear(num_classes, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, num_classes)
                ) for i in range(num_clusters)
            })

    def forward(self, x, cluster_id=None):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # SqueezeNet forward produces N x C x 13 x 13 before final pooling; make it flat
        x = self.net.features(x)
        x = self.net.classifier(x)              # N x num_classes x H x W
        x = self.avgpool(x)                     # N x num_classes x 1 x 1
        x = torch.flatten(x, 1)                 # N x num_classes

        if self.num_clusters == 1:
            return self.head(x)
        if cluster_id is not None:
            return self.heads[f"head_{cluster_id}"](x)
        return {f"head_{i}": head(x) for i, head in self.heads.items()}

def get_client_model(rnd:Random = None):
    if experiment_config.client_net_type == NetType.ALEXNET:
        return AlexNet(num_classes=experiment_config.num_classes).to(device)
    if experiment_config.client_net_type == NetType.VGG:
        return VGGServer(num_classes=experiment_config.num_classes).to(device)
    if experiment_config.client_net_type == NetType.MobileNetV2:
        return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device)

    if experiment_config.client_net_type == NetType.rnd_net:
        p = rnd.random()
        if p<=0.25:
            print("ResNet18Server")
            return ResNet18Server(num_classes=experiment_config.num_classes).to(device)
        if 0.25<p<=0.5:
            print("MobileNetV2Server")

            return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device)
        if 0.5<p<=0.75:
            print("SqueezeNetServer")

            return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device)
        else:
            print("AlexNet")

            return  AlexNet(num_classes=experiment_config.num_classes).to(device)



def get_server_model():
    if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
        num_heads = experiment_config.num_clusters
        if isinstance(num_heads, str):
            num_heads = experiment_config.number_of_optimal_clusters
    else:
        num_heads = 1

    if experiment_config.server_net_type == NetType.ALEXNET:
        return AlexNet(num_classes=experiment_config.num_classes, num_clusters=num_heads).to(device)

    if experiment_config.server_net_type == NetType.VGG:
        return VGGServer(num_classes=experiment_config.num_classes, num_clusters=num_heads).to(device)

    #if experiment_config.server_net_type == NetType.DenseNetServer:
    #    return DenseNetServer(num_classes=experiment_config.num_classes, num_clusters=num_heads).to(device)


class LearningEntity(ABC):
    def __init__(self, id_, global_data, test_global_data):
        self.test_global_data = test_global_data
        self.seed = experiment_config.seed_num
        self.global_data = global_data
        self.pseudo_label_received = {}
        self.pseudo_label_to_send = None
        self.current_iteration = 0
        self.epoch_count = 0
        self.id_ = id_
        self.model = None
        self.accuracy_per_client_1 = {}
        self.accuracy_per_client_10 = {}
        self.accuracy_per_client_100 = {}
        self.accuracy_per_client_5 = {}
        self.size_sent = {}

    # ---------- helpers ----------
    @staticmethod
    def _normalize_probs(p: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(1e-8)
        return p / p.sum(dim=1, keepdim=True)

    @staticmethod
    def _mean_entropy(p: torch.Tensor) -> float:
        p = p.clamp_min(1e-8)
        return float((-(p * p.log()).sum(dim=1)).mean().item())

    def initialize_weights(self, layer):
        self.seed = int((self.seed + 1) % (2**31))
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def get_pseudo_label_L2(self, pseudo_labels):
        loader = DataLoader(self.global_data, batch_size=len(self.global_data))
        _, Y_tensor = next(iter(loader))
        gt = Y_tensor.numpy()
        num_classes = pseudo_labels.shape[1]
        gt_onehot = F.one_hot(torch.tensor(gt), num_classes=num_classes).float().numpy()
        return np.mean(np.linalg.norm(pseudo_labels - gt_onehot, axis=1) ** 2)

    def iterate(self, t):
        if isinstance(self.id_,str):
            torch.manual_seed( t * 17)
            torch.cuda.manual_seed( t * 17)
        else:
            torch.manual_seed(self.id_ + t * 17)
            torch.cuda.manual_seed(self.id_ + t * 17)

        if experiment_config.is_with_memory_load and self.id_ != "server":
            if t == 0:
                self.iteration_context(t)
            else:
                self.model = get_client_model(self.rnd_net)
                self.model.load_state_dict(torch.load(f"./models/model_{self.id_}.pth"))
                self.iteration_context(t)
            torch.save(self.model.state_dict(), f"./models/model_{self.id_}.pth")
            del self.model
        else:
            self.iteration_context(t)

    @abstractmethod
    def iteration_context(self, t):
        pass

    # ---------- evaluation ----------
    def evaluate(self, model=None):
        model = self.model if model is None else model
        global_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for x, _ in global_loader:
                x = x.to(device)
                probs = F.softmax(model(x), dim=1)
                all_probs.append(probs.cpu())
        return torch.cat(all_probs, dim=0)

    def evaluate_max_accuracy_per_point(self, models, data_, k=1, cluster_id=None):
        assert k == 1, "Only top-1 supported here."
        loader = DataLoader(data_, batch_size=1, shuffle=False)
        total = len(data_)
        mat = torch.zeros((len(models), total), dtype=torch.bool)
        for mi, m in enumerate(models):
            m.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(loader):
                    x, y = x.to(device), y.to(device)
                    out = m(x, cluster_id=cluster_id)
                    if out.dim() == 1: out = out.unsqueeze(0)
                    pred = out.argmax(dim=1)
                    mat[mi, i] = (pred == y).item()
        avg = mat.any(dim=0).float().mean().item() * 100
        print(f"Average max accuracy across models: {avg:.2f}%")
        return avg

    def evaluate_accuracy_single(self, data_, model=None, k=1, cluster_id=None):
        model = self.model if model is None else model
        model.eval()
        correct, total = 0, 0
        loader = DataLoader(data_, batch_size=experiment_config.batch_size, shuffle=False)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x, cluster_id=cluster_id)
                if out.dim() == 1: out = out.unsqueeze(0)
                pred = out.argmax(dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        acc = 100 * correct / total if total else 0.0
        print(f"Accuracy for cluster {cluster_id if cluster_id is not None else 'default'}: {acc:.2f}%")
        return acc

    def evaluate_accuracy(self, data_, model=None, k=1, cluster_id=None):
        model = self.model if model is None else model
        model.eval()
        correct, total = 0, 0
        loader = DataLoader(data_, batch_size=experiment_config.batch_size, shuffle=False)
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x, cluster_id=cluster_id)
                if out.dim() == 1: out = out.unsqueeze(0)
                if out.size(1) < k: return 0.0
                _, topk = out.topk(k, dim=1)
                correct += (topk == y.unsqueeze(1)).any(dim=1).sum().item()
                total += y.size(0)
        acc = 100 * correct / total if total else 0.0
        print(f"Top-{k} Accuracy for cluster {cluster_id if cluster_id is not None else 'default'}: {acc:.2f}%")
        return acc

    def evaluate_test_loss(self, cluster_id=None, model=None):
        model = self.model if model is None else model
        model.eval()
        loader = DataLoader(self.test_set, batch_size=experiment_config.batch_size, shuffle=True)
        crit = nn.CrossEntropyLoss()
        tot_loss, tot_n = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x, cluster_id=cluster_id)
                loss = crit(out, y)
                tot_loss += loss.item() * x.size(0)
                tot_n += x.size(0)
        avg = tot_loss / tot_n if tot_n else float('inf')
        print(f"Iteration [{self.current_iteration}], Test Loss (Cluster {cluster_id}): {avg:.4f}")
        return avg


# add at top of your file if not present
import os

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

# assumes: get_client_model, Random, device, experiment_config, InputConsistency exist


class Client(LearningEntity):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, global_data, global_test_data)
        self.num = (self.id_ + 1) * 17
        self.local_test_set = local_test_data
        self.rnd_net = Random((self.seed + 1) * 17 + 13 + (id_ + 1) * 17)
        self.local_data = client_data

        # model once here; weights can be reinitialized later (t==1)
        self.model = get_client_model(self.rnd_net)
        self.model.apply(self.initialize_weights)

        # --- PERSISTENT OPTIMIZERS (created once, reused every round) ---
        self.opt_kd_client = torch.optim.AdamW(
            self.model.parameters(),
            lr=experiment_config.learning_rate_train_c,
            weight_decay=5e-4
        )
        self.opt_finetune_client = torch.optim.Adam(
            self.model.parameters(),
            lr=experiment_config.learning_rate_fine_tune_c
        )
        # ---------------------------------------------------------------

        self.server = None
        self.pseudo_label_L2 = {}
        self.global_label_distribution = self.get_label_distribution()

    # ---------- helpers ----------
    def _ensure_client_optimizers(self, reset: bool = False):
        """
        Re-create optimizers if:
        - reset=True (e.g., after reinit weights at t==1), or
        - param groups are stale (different model object), or
        - optimizer objects are missing.
        """
        if reset or self.opt_kd_client is None:
            self.opt_kd_client = torch.optim.AdamW(
                self.model.parameters(),
                lr=experiment_config.learning_rate_train_c,
                weight_decay=5e-4
            )
        else:
            # replace param groups if they don't point to current model params
            self.opt_kd_client.param_groups[0]['params'] = list(self.model.parameters())

        if reset or self.opt_finetune_client is None:
            self.opt_finetune_client = torch.optim.Adam(
                self.model.parameters(),
                lr=experiment_config.learning_rate_fine_tune_c
            )
        else:
            self.opt_finetune_client.param_groups[0]['params'] = list(self.model.parameters())

    def get_label_distribution(self):
        label_counts = defaultdict(int)
        for _, label in self.local_data:
            label_counts[int(label.item() if hasattr(label, 'item') else int(label))] += 1
        return dict(label_counts)

    # ---------- training (KD, optional consistency/weights) ----------
    def _kd_epoch(self, loader, pseudo_targets_all, optimizer, T, extra_loss_fn=None):
        self.model.train()
        offset = 0
        epoch_loss = 0.0
        for batch_idx, (inputs, y_true) in enumerate(loader):
            bsz = inputs.size(0)
            inputs = inputs.to(device); y_true = y_true.to(device)
            optimizer.zero_grad()

            outputs = self.model(inputs)
            logp = F.log_softmax(outputs / T, dim=1)

            pseudo_targets = pseudo_targets_all[offset:offset + bsz].to(device)
            offset += bsz

            if pseudo_targets.size(0) != bsz:
                print(f"Skipping batch {batch_idx}: Pseudo target size mismatch.")
                continue
            if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                print(f"NaN/Inf in pseudo targets at batch {batch_idx}")
                continue

            pseudo_targets = self._normalize_probs(pseudo_targets)
            loss = F.kl_div(logp, pseudo_targets, reduction='batchmean') * (T * T)

            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(inputs, outputs, y_true, logp, pseudo_targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at batch {batch_idx}: {loss}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / max(1, len(loader))

    def train(self, mean_pseudo_labels):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train ***")

        loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)

        T = getattr(experiment_config, "kd_temperature", 2.0)
        pseudo_targets_all = mean_pseudo_labels.to(device)
        print(f"Teacher PL mean entropy: {self._mean_entropy(pseudo_targets_all):.3f}")

        last = 0.0
        for ep in range(experiment_config.epochs_num_train_client):
            self.epoch_count += 1
            last = self._kd_epoch(loader, pseudo_targets_all, self.opt_kd_client, T)
            print(f"Epoch [{ep + 1}/{experiment_config.epochs_num_train_client}], Loss: {last:.4f}")
        return last

    def train_with_consistency(self, mean_pseudo_labels):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train ***")

        loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)

        T = getattr(experiment_config, "kd_temperature", 2.0)
        lam = experiment_config.lambda_consistency
        pseudo_targets_all = mean_pseudo_labels.to(device)
        print(f"Teacher PL mean entropy: {self._mean_entropy(pseudo_targets_all):.3f}")

        def add_noise(x, std=0.05):
            n = torch.randn_like(x) * std
            return torch.clamp(x + n, 0., 1.)
        mse = nn.MSELoss()

        def extra(inputs, outputs, y_true, logp, pseudo_targets):
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                probs_aug = F.softmax(self.model(add_noise(inputs)), dim=1)
            return lam * mse(probs, probs_aug)

        last = 0.0
        for ep in range(experiment_config.epochs_num_train_client):
            self.epoch_count += 1
            last = self._kd_epoch(loader, pseudo_targets_all, self.opt_kd_client, T, extra_loss_fn=extra)
            print(f"Epoch [{ep + 1}/{experiment_config.epochs_num_train_client}], Loss: {last:.4f}")
        return last

    def train_with_consistency_and_weights(self, mean_pseudo_labels):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train ***")

        loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)

        T = getattr(experiment_config, "kd_temperature", 2.0)
        lam = experiment_config.lambda_consistency
        pseudo_targets_all = mean_pseudo_labels.to(device)
        print(f"Teacher PL mean entropy: {self._mean_entropy(pseudo_targets_all):.3f}")

        present_w, absent_w = 1.0, 0.3
        mse = nn.MSELoss()

        def add_noise(x, std=0.05):
            n = torch.randn_like(x) * std
            return torch.clamp(x + n, 0., 1.)

        def extra(inputs, outputs, y_true, logp, pseudo_targets):
            present = torch.tensor(
                [1.0 if self.global_label_distribution.get(int(lbl.item()), 0) > 0 else 0.0 for lbl in y_true],
                device=device
            )
            weights = present * present_w + (1.0 - present) * absent_w  # [B]
            per_sample_kld = F.kl_div(logp, pseudo_targets, reduction='none').sum(dim=1) * (T * T)
            loss_kd_weighted = (per_sample_kld * weights).mean()

            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                probs_aug = F.softmax(self.model(add_noise(inputs)), dim=1)
            loss_cons = mse(probs, probs_aug)

            # replace base KD with weighted KD, still add consistency
            return (loss_kd_weighted - per_sample_kld.mean()) + lam * loss_cons

        last = 0.0
        for ep in range(experiment_config.epochs_num_train_client):
            self.epoch_count += 1
            last = self._kd_epoch(loader, pseudo_targets_all, self.opt_kd_client, T, extra_loss_fn=extra)
            print(f"Epoch [{ep + 1}/{experiment_config.epochs_num_train_client}], Loss: {last:.4f}")
        return last

    # ---------- orchestration ----------
    def iteration_context(self, t):
        self.current_iteration = t

        # (A) If t==1, reinitialize weights ONCE (per your request) and reset optimizers
        if t == 1:
            self.model.apply(self.initialize_weights)
            self._ensure_client_optimizers(reset=True)

        has_server_pl = isinstance(self.pseudo_label_received, torch.Tensor)

        # (B) KD only after server feedback (round >= 2)
        if t > 1 and has_server_pl:
            if experiment_config.input_consistency == InputConsistency.withInputConsistency:
                if experiment_config.weights_for_ps:
                    _ = self.train_with_consistency_and_weights(self.pseudo_label_received)
                else:
                    _ = self.train_with_consistency(self.pseudo_label_received)
            else:
                if experiment_config.weights_for_ps:
                    _ = self.train_with_weights(self.pseudo_label_received)
                else:
                    _ = self.train(self.pseudo_label_received)

        # (C) Always do local fine-tune each round
        if t == 0:
            # first contact: optimizers already exist
            self._ensure_client_optimizers(reset=False)
            self.fine_tune(50)
        else:
            # subsequent rounds: do not reinit weights (except t==1 handled above)
            self._ensure_client_optimizers(reset=False)
            self.fine_tune()

        # send PLs to server + stats
        self.pseudo_label_to_send = self.evaluate()
        pl = self.pseudo_label_to_send
        self.size_sent[t] = (pl.numel() * pl.element_size()) / (1024 * 1024)
        self.pseudo_label_L2[t] = self.get_pseudo_label_L2(pl)

        self.accuracy_per_client_1[t] = self.evaluate_accuracy_single(self.local_test_set, k=1)
        self.accuracy_per_client_10[t] = self.evaluate_accuracy(self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)
        self.accuracy_per_client_5[t] = self.evaluate_accuracy(self.local_test_set, k=5)
        print("hi")

    # If you need the simple “weights only” KD:
    def train_with_weights(self, mean_pseudo_labels):
        return self.train(mean_pseudo_labels)

    def __str__(self):
        return "Client " + str(self.id_)

    def fine_tune(self, num_of_epochs=experiment_config.epochs_num_input_fine_tune_clients):
        print("*** " + self.__str__() + " fine-tune ***")
        loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size, shuffle=True)
        self.model.train()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        last = 0.0
        for ep in range(num_of_epochs):
            self.epoch_count += 1
            epoch_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                self.opt_finetune_client.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                self.opt_finetune_client.step()
                epoch_loss += loss.item()
            last = epoch_loss / max(1, len(loader))
            print(f"Epoch [{ep + 1}/{num_of_epochs}], Loss: {last:.4f}")
        return last

    def print_grad_size(self):
        total_elems = 0
        total_bytes = 0
        norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                total_elems += p.grad.numel()
                total_bytes += p.grad.numel() * p.grad.element_size()
                norms.append(p.grad.detach().norm(2))
        if norms:
            total_l2 = torch.norm(torch.stack(norms), 2).item()
            print(f"Total gradient L2 norm: {total_l2:.4f}")
        print(f"Total gradient elements: {total_elems}")
        print(f"Total gradient size: {total_bytes / 1024 / 1024:.4f} MB\n")

    # ---------- optional: persist optimizer state when using "memory load" ----------
    def iterate(self, t):
        # override to ALSO save/load optimizer state when is_with_memory_load is True
        if isinstance(self.id_, str):
            torch.manual_seed(t * 17)
            torch.cuda.manual_seed(t * 17)
        else:
            torch.manual_seed(self.id_ + t * 17)
            torch.cuda.manual_seed(self.id_ + t * 17)

        if experiment_config.is_with_memory_load:
            if t == 0:
                self.iteration_context(t)
            else:
                # rebuild model, then load weights
                self.model = get_client_model(self.rnd_net)
                self.model.load_state_dict(torch.load(f"./models/model_{self.id_}.pth"))
                # ensure optimizers exist and point to correct params
                self._ensure_client_optimizers(reset=False)
                # load optimizer states if exist
                opt_kd_path = f"./models/opt_kd_{self.id_}.pth"
                opt_ft_path = f"./models/opt_ft_{self.id_}.pth"
                if os.path.exists(opt_kd_path):
                    self.opt_kd_client.load_state_dict(torch.load(opt_kd_path))
                if os.path.exists(opt_ft_path):
                    self.opt_finetune_client.load_state_dict(torch.load(opt_ft_path))
                self.iteration_context(t)

            # save model + optimizer states
            os.makedirs("./models", exist_ok=True)
            torch.save(self.model.state_dict(), f"./models/model_{self.id_}.pth")
            torch.save(self.opt_kd_client.state_dict(), f"./models/opt_kd_{self.id_}.pth")
            torch.save(self.opt_finetune_client.state_dict(), f"./models/opt_ft_{self.id_}.pth")
        else:
            self.iteration_context(t)


class Server(LearningEntity):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict):
        super().__init__(id_, global_data, test_data)
        self.pseudo_label_before_net_L2 = {}
        self.pseudo_label_after_net_L2 = {}
        self.num = 1000 * 17
        self.pseudo_label_received = {}
        self.clusters_client_id_dict_per_iter = {}
        self.clients_ids = clients_ids
        self.reset_clients_received_pl()
        self.clients_test_data_dict = clients_test_data_dict

        # previous kmeans centroids (if used)
        self.previous_centroids_dict = {}
        if isinstance(experiment_config.num_clusters, int):
            num_clusters = experiment_config.num_clusters
        else:
            num_clusters = experiment_config.number_of_optimal_clusters
        for c in range(num_clusters):
            self.previous_centroids_dict[c] = None

        # --- MODEL(S) + PERSISTENT OPTIMIZER(S) ---
        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
            self.model = get_server_model()
            self.model.apply(self.initialize_weights)
            self.server_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=experiment_config.learning_rate_train_s,
                weight_decay=5e-4
            )
            self.multi_model_dict = None
            self.server_opt_dict = None

        elif experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            self.model = None
            self.multi_model_dict = {}
            self.server_opt_dict = {}
            for c in range(num_clusters):
                m = get_server_model()
                m.apply(self.initialize_weights)
                self.multi_model_dict[c] = m
                self.server_opt_dict[c] = torch.optim.AdamW(
                    m.parameters(),
                    lr=experiment_config.learning_rate_train_s,
                    weight_decay=5e-4
                )
        # ------------------------------------------------

        self.accuracy_per_client_1_max = {}
        self.accuracy_per_client_10_max = {}
        self.accuracy_per_client_100_max = {}
        self.accuracy_per_client_5_max = {}
        self.accuracy_server_test_1 = {}
        self.accuracy_global_data_1 = {}

        for cid in self.clients_ids:
            self.accuracy_per_client_1[cid] = {}
            self.accuracy_per_client_1_max[cid] = {}
            self.accuracy_per_client_5_max[cid] = {}
            self.accuracy_per_client_10_max[cid] = {}
            self.accuracy_per_client_100_max[cid] = {}

        for c in range(num_clusters):
            self.accuracy_server_test_1[c] = {}
            self.accuracy_global_data_1[c] = {}

    # ---------- comms ----------
    def receive_single_pseudo_label(self, sender, info):
        self.pseudo_label_received[sender] = info

    # ---------- PL routing ----------
    def get_pseudo_label_list_after_models_train(self, mean_pseudo_labels_per_cluster):
        out = {}
        for cluster_id, pl in mean_pseudo_labels_per_cluster.items():
            model_ = self.model_per_cluster[cluster_id]
            self.train(pl, cluster_id, model_)
            out[cluster_id] = self.evaluate(model_)
        return list(out.values())

    @staticmethod
    def select_confident_pseudo_labels(cluster_pseudo_labels):
        num_points = cluster_pseudo_labels[0].size(0)
        max_conf = torch.zeros(num_points, device=cluster_pseudo_labels[0].device)
        selected = torch.zeros_like(cluster_pseudo_labels[0])
        for pl in cluster_pseudo_labels:
            cmax, _ = torch.max(pl, dim=1)
            mask = cmax > max_conf
            max_conf[mask] = cmax[mask]
            selected[mask] = pl[mask]
        return selected

    def _get_server_optimizer(self, cluster_num, selected_model):
        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            # cluster_num is an int index in this mode
            return self.server_opt_dict[int(cluster_num)]
        else:
            return self.server_optimizer

    def create_feed_back_to_clients_multihead(self, mean_pseudo_labels_per_cluster, t):
        for _ in range(experiment_config.num_rounds_multi_head):
            for cluster_id, pl in mean_pseudo_labels_per_cluster.items():
                self.train(pl, cluster_id)
            if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_cluster:
                for cluster_id in mean_pseudo_labels_per_cluster.keys():
                    cluster_pl = self.evaluate_for_cluster(cluster_id)
                    for client_id in self.clusters_client_id_dict_per_iter[t][cluster_id]:
                        self.pseudo_label_to_send[client_id] = cluster_pl

        if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_client:
            out = []
            for cluster_id, pl in mean_pseudo_labels_per_cluster.items():
                self.train(pl, cluster_id)
                out.append(self.evaluate_for_cluster(cluster_id))
            merged = self.select_confident_pseudo_labels(out)
            for client_id in self.clients_ids:
                self.pseudo_label_to_send[client_id] = merged

    def create_feed_back_to_clients_multimodel(self, mean_pseudo_labels_per_cluster, t):
        if not isinstance(self.pseudo_label_to_send, dict):
            self.pseudo_label_to_send = {}

        pl_per_cluster = {}

        for cluster_id, mean_pl in mean_pseudo_labels_per_cluster.items():
            selected_model = self.multi_model_dict[cluster_id]

            for _ in range(5):
                if experiment_config.input_consistency == InputConsistency.withInputConsistency:
                    self.train_with_consistency(mean_pl, cluster_id, selected_model)
                else:
                    self.train(mean_pl, cluster_id, selected_model)

                acc_global = self.evaluate_accuracy_single(self.test_global_data, model=selected_model, k=1, cluster_id=0)
                acc_local = self.evaluate_accuracy_single(self.global_data,     model=selected_model, k=1, cluster_id=0)
                # (keep your break/reinit logic here if you had one)

            if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_cluster:
                cluster_pl = self.evaluate_for_cluster(0, selected_model)
                pl_per_cluster[cluster_id] = cluster_pl
                for client_id in self.clusters_client_id_dict_per_iter[t][cluster_id]:
                    self.pseudo_label_to_send[client_id] = cluster_pl

        if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_client:
            cluster_pl_list = []
            for cluster_id, mean_pl in mean_pseudo_labels_per_cluster.items():
                selected_model = self.multi_model_dict[cluster_id]
                self.train(mean_pl, cluster_id, selected_model)
                pl = self.evaluate_for_cluster(0, selected_model)
                pl_per_cluster[cluster_id] = pl
                cluster_pl_list.append(pl)

            merged = self.select_confident_pseudo_labels(cluster_pl_list)
            for client_id in self.clients_ids:
                self.pseudo_label_to_send[client_id] = merged

        return pl_per_cluster

    # ---------- evaluation across clusters/clients ----------
    def evaluate_results(self, t):
        if isinstance(experiment_config.num_clusters, int):
            num_clusters = experiment_config.num_clusters
        else:
            num_clusters = experiment_config.number_of_optimal_clusters

        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            models_list = list(self.multi_model_dict.values())
        else:
            models_list = [self.model]

        self.accuracy_global_data_1[t] = self.evaluate_max_accuracy_per_point(models=models_list,
                                                                              data_=self.global_data, k=1,
                                                                              cluster_id=None)
        for cluster_id in range(num_clusters):
            selected_model = None
            cluster_id_to_examine = cluster_id
            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                selected_model = self.multi_model_dict[cluster_id]
                cluster_id_to_examine = 0
            self.accuracy_server_test_1[cluster_id][t] = self.evaluate_accuracy_single(
                self.test_global_data, model=selected_model, k=1, cluster_id=cluster_id_to_examine
            )

        for client_id in self.clients_ids:
            test_data_per_clients = self.clients_test_data_dict[client_id]
            cluster_id_for_client = self.get_cluster_of_client(client_id, t)

            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                selected_model = self.multi_model_dict[cluster_id_for_client]
                cluster_for_eval = 0
            else:
                selected_model = None
                cluster_for_eval = cluster_id_for_client

            print("client_id", client_id, "accuracy_per_client_1")
            self.accuracy_per_client_1[client_id][t] = self.evaluate_accuracy_single(
                test_data_per_clients, model=selected_model, k=1, cluster_id=cluster_for_eval
            )

            l1, l2, l3, l4 = [], [], [], []
            for c in range(num_clusters):
                if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                    m = self.multi_model_dict[c]
                    l1.append(self.evaluate_accuracy_single(test_data_per_clients, model=m, k=1,   cluster_id=0))
                    l2.append(self.evaluate_accuracy(        test_data_per_clients, model=m, k=10,  cluster_id=0))
                    l3.append(self.evaluate_accuracy(        test_data_per_clients, model=m, k=100, cluster_id=0))
                    l4.append(self.evaluate_accuracy(        test_data_per_clients, model=m, k=5,   cluster_id=0))
                else:
                    l1.append(self.evaluate_accuracy(test_data_per_clients, model=selected_model, k=1, cluster_id=c))
            print("client_id", client_id, "accuracy_per_client_1_max", max(l1))

            self.accuracy_per_client_1_max[client_id][t]   = max(l1)
            if l2: self.accuracy_per_client_10_max[client_id][t]  = max(l2)
            if l3: self.accuracy_per_client_100_max[client_id][t] = max(l3)
            if l4: self.accuracy_per_client_5_max[client_id][t]   = max(l4)

    # ---------- round orchestration ----------
    def iteration_context(self, t):
        self.current_iteration = t

        # Reset server outbox every round
        self.pseudo_label_to_send = {}

        mean_pl_per_cluster, self.clusters_client_id_dict_per_iter[t] = \
            self.get_pseudo_labels_input_per_cluster(t)

        self.pseudo_label_before_net_L2[t] = {}
        if t > 0:
            for cid, pl in mean_pl_per_cluster.items():
                self.pseudo_label_before_net_L2[t][cid] = self.get_pseudo_label_L2(pl)

            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
                self.create_feed_back_to_clients_multihead(mean_pl_per_cluster, t)
            elif experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                _ = self.create_feed_back_to_clients_multimodel(mean_pl_per_cluster, t)

            self.pseudo_label_after_net_L2[t] = 0
            self.evaluate_results(t)

        self.reset_clients_received_pl()

    # ---------- server KD ----------
    def train_with_consistency(self, mean_pseudo_labels, cluster_num="0", selected_model=None):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train *** Cluster: {cluster_num} ***")

        loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)

        m = self.model if selected_model is None else selected_model
        m.train()
        T = getattr(experiment_config, "kd_temperature", 2.0)
        lam = experiment_config.lambda_consistency
        optimizer = self._get_server_optimizer(cluster_num, m)
        pseudo_targets_all = mean_pseudo_labels.to(device)
        print(f"Teacher PL mean entropy: {self._mean_entropy(pseudo_targets_all):.3f}")

        def add_noise(x, std=0.05):
            n = torch.randn_like(x) * std
            return torch.clamp(x + n, 0., 1.)
        mse = nn.MSELoss()

        last = 0.0
        for ep in range(experiment_config.epochs_num_train_server):
            self.epoch_count += 1
            offset = 0
            epoch_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs = m(inputs, cluster_id=cluster_num)
                logp = F.log_softmax(outputs / T, dim=1)

                bsz = inputs.size(0)
                pseudo_targets = pseudo_targets_all[offset:offset + bsz].to(device)
                offset += bsz
                if pseudo_targets.size(0) != bsz:
                    print(f"Skipping batch {batch_idx}: Pseudo target size mismatch.")
                    continue

                pseudo_targets = self._normalize_probs(pseudo_targets)
                loss_kd = F.kl_div(logp, pseudo_targets, reduction='batchmean') * (T * T)

                with torch.no_grad():
                    probs     = F.softmax(outputs, dim=1)
                    probs_aug = F.softmax(m(add_noise(inputs), cluster_id=cluster_num), dim=1)
                loss_cons = mse(probs, probs_aug)

                loss = loss_kd + lam * loss_cons
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            last = epoch_loss / max(1, len(loader))
            print(f"Epoch [{ep + 1}/{experiment_config.epochs_num_train_server}], Loss: {last:.4f}")
        return last

    def train(self, mean_pseudo_labels, cluster_num="0", selected_model=None):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train *** Cluster: {cluster_num} ***")

        loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)

        m = self.model if selected_model is None else selected_model
        m.train()
        T = getattr(experiment_config, "kd_temperature", 2.0)
        optimizer = self._get_server_optimizer(cluster_num, m)
        pseudo_targets_all = mean_pseudo_labels.to(device)
        print(f"Teacher PL mean entropy: {self._mean_entropy(pseudo_targets_all):.3f}")

        last = 0.0
        for ep in range(experiment_config.epochs_num_train_server):
            self.epoch_count += 1
            offset = 0
            epoch_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs = m(inputs, cluster_id=cluster_num)
                logp = F.log_softmax(outputs / T, dim=1)

                bsz = inputs.size(0)
                pseudo_targets = pseudo_targets_all[offset:offset + bsz].to(device)
                offset += bsz
                if pseudo_targets.size(0) != bsz:
                    print(f"Skipping batch {batch_idx}: Expected pseudo target size {bsz}, got {pseudo_targets.size(0)}")
                    continue

                pseudo_targets = self._normalize_probs(pseudo_targets)
                loss = F.kl_div(logp, pseudo_targets, reduction='batchmean') * (T * T)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            last = epoch_loss / max(1, len(loader))
            print(f"Epoch [{ep + 1}/{experiment_config.epochs_num_train_server}], Loss: {last:.4f}")
        return last

    # ---------- utils (unchanged logic, kept for completeness) ----------
    def reset_clients_received_pl(self):
        for id_ in self.clients_ids:
            self.pseudo_label_received[id_] = None

    def k_means_grouping(self):
        k = experiment_config.num_clusters
        client_ids = list(self.pseudo_label_received.keys())
        pseudo_labels = [self.pseudo_label_received[cid].flatten().numpy() for cid in client_ids]
        data_matrix = np.vstack(pseudo_labels)
        if not self.centroids_are_empty():
            prev = np.array(list(self.previous_centroids_dict.values()))
            if prev.shape[0] != k:
                raise ValueError(f"Previous centroids count does not match k={k}.")
            kmeans = KMeans(n_clusters=k, init=prev, n_init=1, random_state=42)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_matrix)
        self.previous_centroids_dict = {i: c for i, c in enumerate(kmeans.cluster_centers_)}
        assigns = kmeans.predict(data_matrix)
        clusters = {i: [] for i in range(k)}
        for cid, cl in zip(client_ids, assigns):
            clusters[cl].append(cid)
        return clusters

    def get_cluster_mean_pseudo_labels_dict(self, clusters_client_id_dict):
        ans = {}
        for cluster_id, clients_ids in clusters_client_id_dict.items():
            ans[cluster_id] = [self.pseudo_label_received[c] for c in clients_ids]
        return ans

    def calc_L2(self, pair):
        return Server.calc_L2_given_pls(self.pseudo_label_received[pair[0]],
                                        self.pseudo_label_received[pair[1]])

    @staticmethod
    def calc_L2_given_pls(pl1, pl2):
        return torch.sqrt(torch.sum((pl1 - pl2) ** 2)).item()

    def initiate_clusters_centers_dict(self, L2_of_all_clients):
        max_pair = max(L2_of_all_clients.items(), key=lambda item: item[1])
        k1, k2 = max_pair[0]
        return {k1: self.pseudo_label_received[k1], k2: self.pseudo_label_received[k2]}

    def update_distance_of_all_clients(self, L2_of_all_clients, clusters_centers_dict):
        L2_temp = {}
        for (id1, id2), v in L2_of_all_clients.items():
            if (id1 in clusters_centers_dict) ^ (id2 in clusters_centers_dict):
                L2_temp[(id1, id2)] = v
        return L2_temp

    def get_l2_of_non_centers(self, L2_of_all_clients, clusters_centers_dict):
        all_vals = {}
        for (id1, id2), l2 in L2_of_all_clients.items():
            if id1 in clusters_centers_dict:
                which = id2
            elif id2 in clusters_centers_dict:
                which = id1
            else:
                continue
            all_vals.setdefault(which, []).append(l2)
        sums = {k: sum(v) for k, v in all_vals.items()}
        return max(sums, key=sums.get)

    def complete_clusters_centers_and_L2_of_all_clients(self, clusters_centers_dict):
        cluster_counter = experiment_config.num_clusters - 2 if isinstance(experiment_config.num_clusters, int) else 3
        while cluster_counter > 0:
            dist_all = self.get_distance_dict()
            dist_all = self.update_distance_of_all_clients(dist_all, clusters_centers_dict)
            new_center = self.get_l2_of_non_centers(dist_all, clusters_centers_dict)
            clusters_centers_dict[new_center] = self.pseudo_label_received[new_center]
            cluster_counter -= 1
        dist_all = self.get_distance_dict()
        dist_all = self.update_distance_of_all_clients(dist_all, clusters_centers_dict)
        return dist_all, clusters_centers_dict

    def get_l2_of_non_center_to_center(self, L2_from_center_clients, clusters_centers_dict):
        ans = {}
        for (id1, id2), l2 in L2_from_center_clients.items():
            if id1 in clusters_centers_dict:
                not_center, center = id2, id1
            elif id2 in clusters_centers_dict:
                not_center, center = id1, id2
            else:
                continue
            ans.setdefault(not_center, {})[center] = l2
        return ans

    def get_non_center_to_which_center_dict(self, l2_non_center_to_center):
        one_to_one = {nc: min(d, key=d.get) for nc, d in l2_non_center_to_center.items()}
        inv = {}
        for k, v in one_to_one.items():
            inv.setdefault(v, []).append(k)
        return inv

    def prep_clusters(self, mapping):
        ans, counter = {}, 0
        for center, others in mapping.items():
            ans[counter] = [center] + list(others)
            counter += 1
        return ans

    def get_clusters_centers_dict(self):
        L2_all = self.get_distance_dict()
        centers = self.initiate_clusters_centers_dict(L2_all)
        L2_from_centers, centers = self.complete_clusters_centers_and_L2_of_all_clients(centers)
        non_center_map = self.get_l2_of_non_center_to_center(L2_from_centers, centers)
        mapping = self.get_non_center_to_which_center_dict(non_center_map)
        ans = self.prep_clusters(mapping)
        to_add = self.get_centers_to_add(centers, ans)
        for idx, c in enumerate(to_add, start=max(ans.keys(), default=-1) + 1):
            ans[idx] = [c]
        return ans

    def get_centers_to_add(self, centers, ans):
        lst = []
        for center in centers.keys():
            if self.center_id_not_in_ans(ans, center) is not None:
                lst.append(center)
        return lst

    def center_id_not_in_ans(self, ans, center_id):
        for lst in ans.values():
            if center_id in lst:
                return None
        return center_id

    def manual_grouping(self):
        clusters = {}
        if isinstance(experiment_config.num_clusters, int):
            k = experiment_config.num_clusters
        else:
            k = experiment_config.number_of_optimal_clusters
        if k == 1:
            clusters[0] = self.clients_ids
        else:
            clusters = self.get_clusters_centers_dict()
        return clusters

    def get_pseudo_labels_input_per_cluster(self, timestamp):
        mean_per_cluster = {}
        flag = False
        clusters = None

        if experiment_config.num_clusters == "Optimal":
            clusters = experiment_config.known_clusters; flag = True
        if experiment_config.num_clusters == 1:
            clusters = {0: self.clients_ids}; flag = True

        if (experiment_config.cluster_technique in (ClusterTechnique.greedy_elimination_cross_entropy,
                                                    ClusterTechnique.greedy_elimination_L2)) and not flag:
            if timestamp == 0:
                clusters = self.greedy_elimination()
            else:
                clusters = self.clusters_client_id_dict_per_iter[0]

        if experiment_config.cluster_technique == ClusterTechnique.kmeans and not flag:
            clusters = self.k_means_grouping()

        if (experiment_config.cluster_technique in (ClusterTechnique.manual_L2,
                                                    ClusterTechnique.manual_cross_entropy)) and not flag:
            clusters = self.manual_grouping()

        if experiment_config.cluster_technique == ClusterTechnique.manual_single_iter and not flag:
            clusters = self.manual_grouping() if timestamp == 0 else self.clusters_client_id_dict_per_iter[0]

        cluster_to_pl_list = self.get_cluster_mean_pseudo_labels_dict(clusters)
        for cid, pls in cluster_to_pl_list.items():
            if experiment_config.server_input_tech == ServerInputTech.mean:
                mean_per_cluster[cid] = torch.mean(torch.stack(list(pls)), dim=0)
            else:
                mean_per_cluster[cid] = self.select_confident_pseudo_labels(list(pls))
        return mean_per_cluster, clusters

    def evaluate_for_cluster(self, cluster_id, model=None):
        model = self.model if model is None else model
        print(f"*** Evaluating Cluster {cluster_id} Head ***")
        model.eval()
        loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
                    out = model(x, cluster_id=cluster_id)
                else:
                    out = model(x, cluster_id=0)
                all_probs.append(F.softmax(out, dim=1).cpu())
        return torch.cat(all_probs, dim=0)

    def __str__(self):
        return "server"

    def centroids_are_empty(self):
        return any(v is None for v in self.previous_centroids_dict.values())

    def get_cluster_of_client(self, client_id, t):
        if experiment_config.cluster_technique == ClusterTechnique.manual_single_iter:
            for cid, lst in self.clusters_client_id_dict_per_iter[0].items():
                if client_id in lst:
                    return cid
        else:
            for cid, lst in self.clusters_client_id_dict_per_iter[t].items():
                if client_id in lst:
                    return cid

    def init_models_measures(self):
        # called when number of clusters changes (greedy)
        num_clusters = experiment_config.num_clusters
        for cid in range(num_clusters):
            self.previous_centroids_dict[cid] = None
            self.accuracy_server_test_1[cid] = {}
            self.accuracy_global_data_1[cid] = {}

        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            # rebuild models AND persistent optimizers for each cluster
            self.multi_model_dict = {}
            self.server_opt_dict = {}
            for c in range(num_clusters):
                m = get_server_model()
                m.apply(self.initialize_weights)
                self.multi_model_dict[c] = m
                self.server_opt_dict[c] = torch.optim.AdamW(
                    m.parameters(),
                    lr=experiment_config.learning_rate_train_s,
                    weight_decay=5e-4
                )

    def get_distance_per_client(self, distance_dict):
        ans = {}
        for (a, b), d in distance_dict.items():
            ans.setdefault(a, {})[b] = d
            ans.setdefault(b, {})[a] = d
        return ans

    def greedy_elimination(self):
        distance_dict = self.get_distance_dict()
        per_client = self.get_distance_per_client(distance_dict)
        epsilon_ = self.calc_epsilon(per_client)
        clusters = self.greedy_elimination_t0(epsilon_, per_client)
        experiment_config.num_clusters = len(clusters)
        self.init_models_measures()
        return clusters

    def get_distance_dict(self):
        pairs = list(combinations(self.clients_ids, 2))
        out = {}
        for pair in pairs:
            if experiment_config.cluster_technique in (ClusterTechnique.greedy_elimination_L2,
                                                       ClusterTechnique.manual_L2):
                out[pair] = self.calc_L2(pair)
            else:
                out[pair] = self.calc_cross_entropy(pair)
        return out

    def calc_cross_entropy(self, pair):
        return Server.calc_cross_entropy_given_pl(self.pseudo_label_received[pair[0]],
                                                  self.pseudo_label_received[pair[1]])

    @staticmethod
    def calc_cross_entropy_given_pl(first_pl, second_pl):
        loss1 = -(first_pl * torch.log(second_pl.clamp_min(1e-9))).sum(dim=1).mean()
        loss2 = -(second_pl * torch.log(first_pl.clamp_min(1e-9))).sum(dim=1).mean()
        return (0.5 * (loss1 + loss2)).item()

    def get_pseudo_label_in_cluster(self, clusters_client_id_dict):
        d = {}
        for cid, clients in clusters_client_id_dict.items():
            d[cid] = [self.pseudo_label_received[c] for c in clients]
        return d

    def calc_epsilon(self, distance_per_client):
        epsilon_ = 0.1
        while True:
            if len(self.greedy_elimination_t0(epsilon_, copy.deepcopy(distance_per_client))) <= \
               experiment_config.number_of_optimal_clusters + experiment_config.cluster_addition:
                break
            epsilon_ += 0.1
        with open("results.txt", "a") as f:
            f.write(f"BETA for {experiment_config.cluster_addition} for seed {experiment_config.seed_num} is {epsilon_}\n")
        return epsilon_

    def compute_distances(self, id_label_dict, distance_function):
        out = {}
        ids = list(id_label_dict.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id1, id2 = ids[i], ids[j]
                out[(id1, id2)] = distance_function(id_label_dict[id1], id_label_dict[id2])
        return out

    def filter_far_clients(self, distance_per_client, epsilon):
        for client, dd in distance_per_client.items():
            rem = [o for o, dist in dd.items() if dist > epsilon]
            for o in rem:
                del dd[o]

    def greedy_elimination_t0(self, epsilon_, distance_per_client):
        self.filter_far_clients(distance_per_client, epsilon_)
        clusters = {}
        counter = -1
        while len(distance_per_client) > 0:
            counter += 1
            max_client = max(distance_per_client, key=lambda k: len(distance_per_client[k]))
            others = list(distance_per_client[max_client].keys())
            group = copy.deepcopy(others) + [max_client]
            clusters[counter] = group
            for u in group:
                for d in distance_per_client.values():
                    if u in d: del d[u]
                del distance_per_client[u]
        return clusters

class Client_pFedCK(Client):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)
        self.personalized_model = AlexNet(num_classes=experiment_config.num_classes).to(device)
        self.interactive_model = AlexNet(num_classes=experiment_config.num_classes).to(device)
        self.initial_state = None  # To store interactive model's state before training

    def set_models(self, personalized_state, interactive_state):
        self.personalized_model.load_state_dict(personalized_state)
        self.interactive_model.load_state_dict(interactive_state)

    def train(self, t):
        self.current_iteration = t

        # Save initial interactive model state for variation calculation
        self.initial_state = copy.deepcopy(self.interactive_model.state_dict())

        self.personalized_model.train()
        self.interactive_model.train()

        optimizer_personalized = torch.optim.Adam(self.personalized_model.parameters(), lr=1e-4)
        optimizer_interaction = torch.optim.Adam(self.interactive_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            self.local_data,
            batch_size=experiment_config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        print(f"\nClient {self.id_} begins training\n")

        for epoch in range(5):  # or experiment_config.epochs_num_train_client
            total_loss_personalized, total_loss_interactive = 0.0, 0.0
            batch_count = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                batch_count += 1

                # Train personalized model
                optimizer_personalized.zero_grad()
                loss_personalized = criterion(self.personalized_model(x), y)
                loss_personalized.backward()
                optimizer_personalized.step()
                total_loss_personalized += loss_personalized.item()

                # Train interactive model
                optimizer_interaction.zero_grad()
                loss_interactive = criterion(self.interactive_model(x), y)
                loss_interactive.backward()
                optimizer_interaction.step()
                total_loss_interactive += loss_interactive.item()

                # Sync personalized model with updated interactive model
                self.personalized_model.load_state_dict(self.interactive_model.state_dict())

            print(f"Epoch {epoch + 1}/5 | Personalized Loss: {total_loss_personalized / batch_count:.4f} "
                  f"| Interactive Loss: {total_loss_interactive / batch_count:.4f}")

        self.accuracy_per_client_1[t] = self.evaluate_accuracy_single(data_=self.local_test_set,
                                                                      model=self.personalized_model, k=1)
        self.accuracy_per_client_5[t] = self.evaluate_accuracy(data_=self.local_test_set, model=self.personalized_model,
                                                               k=5)
        self.accuracy_per_client_10[t] = self.evaluate_accuracy(data_=self.local_test_set,
                                                                model=self.personalized_model, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(data_=self.local_test_set,
                                                                 model=self.personalized_model, k=100)

        print("accuracy_per_client_1", self.accuracy_per_client_1[t])
        return self.calculate_param_variation()

    def calculate_param_variation(self):
        if self.initial_state is None:
            raise ValueError("Initial state not set. Did you call train()?")

        param_variations = {
            key: self.interactive_model.state_dict()[key] - self.initial_state[key]
            for key in self.initial_state
        }
        return param_variations

    def update_interactive_model(self, avg_param_variations):
        updated_state = self.interactive_model.state_dict()
        for key, variation in avg_param_variations.items():
            if key in updated_state:
                updated_state[key] += variation
        self.interactive_model.load_state_dict(updated_state)
        print(f"Client {self.id_}: Updated interactive model with average parameter variations.")


class Client_FedAvg(Client):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        Client.__init__(self, id_, client_data, global_data, global_test_data, local_test_data)
        self.weights_received = None
        self.weights_to_send = None

    def iteration_context(self, t):

        self.current_iteration = t
        flag = False
        for _ in range(1000):
            if t == 0:
                # self.model.apply(self.initialize_weights)
                # self.model.apply(self.initialize_weights)

                self.model.apply(self.initialize_weights)

            if t > 0:
                if flag:
                    self.model.apply(self.initialize_weights)
                else:
                    # self.model.apply(self.initialize_weights)

                    self.model.load_state_dict(self.weights_received)

            self.weights_to_send = self.fine_tune()

            total_size = 0
            for param in self.weights_to_send.values():
                total_size += param.numel() * param.element_size()
            self.size_sent[t] = total_size / (1024 * 1024)
            acc = self.evaluate_accuracy_single(self.local_test_set)

            acc_test = self.evaluate_accuracy_single(self.test_global_data)
            if experiment_config.data_set_selected == DataSet.CIFAR100:
                if acc_test != 1:
                    break
                else:
                    flag = True

            if experiment_config.data_set_selected == DataSet.CIFAR10 or experiment_config.data_set_selected == DataSet.SVHN:
                if acc != 10 and acc_test != 10:
                    break
                else:
                    flag = True
                    # self.model.apply(self.initialize_weights)
            if experiment_config.data_set_selected == DataSet.TinyImageNet:
                if acc != 0.5 and acc_test != 0.5:
                    break
                else:
                    flag = True
            if experiment_config.data_set_selected == DataSet.EMNIST_balanced:
                if acc > 2.14 and acc_test > 2.14:
                    break
                else:
                    flag = True

        self.accuracy_per_client_1[t] = self.evaluate_accuracy_single(self.local_test_set, k=1)
        self.accuracy_per_client_5[t] = self.evaluate_accuracy(self.local_test_set, k=5)
        self.accuracy_per_client_10[t] = self.evaluate_accuracy(self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)

    def fine_tune(self):
        print("*** " + self.__str__() + " fine-tune ***")

        # Load the weights into the model
        # if self.weights is  None:
        #    self.model.apply(self.initialize_weights)
        # else:
        #    self.model.load_state_dict(self.weights)

        # Create a DataLoader for the local data
        fine_tune_loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size, shuffle=True)
        self.model.train()  # Set the model to training mode

        # Define loss function and optimizer

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_fine_tune_c)

        epochs = experiment_config.epochs_num_input_fine_tune_clients
        for epoch in range(epochs):
            self.epoch_count += 1
            epoch_loss = 0
            for inputs, targets in fine_tune_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            result_to_print = epoch_loss / len(fine_tune_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {result_to_print:.4f}")
        ans = self.model.state_dict()

        return ans


class Client_NoFederatedLearning(Client):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data, evaluate_every):
        Client.__init__(self, id_, client_data, global_data, global_test_data, local_test_data)
        self.evaluate_every = evaluate_every

    def fine_tune(self):
        print("*** " + self.__str__() + " fine-tune ***")

        fine_tune_loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size, shuffle=True)
        self.model.train()  # Set the model to training mode

        # Define loss function and optimizer

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_fine_tune_c)

        epochs = experiment_config.epochs_num_input_fine_tune_clients_no_fl
        for epoch in range(epochs):
            self.epoch_count += 1
            epoch_loss = 0
            for inputs, targets in fine_tune_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % self.evaluate_every == 0 and epoch != 0:
                self.accuracy_per_client_1[epoch] = self.evaluate_accuracy_single(self.local_test_set, k=1)
                self.accuracy_per_client_5[epoch] = self.evaluate_accuracy(self.local_test_set, k=5)
                self.accuracy_per_client_10[epoch] = self.evaluate_accuracy(self.local_test_set, k=10)
                self.accuracy_per_client_100[epoch] = self.evaluate_accuracy(self.local_test_set, k=100)

            result_to_print = epoch_loss / len(fine_tune_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {result_to_print:.4f}")
        # self.weights = self.model.state_dict()self.weights = self.model.state_dict()

        return result_to_print


class Client_PseudoLabelsClusters_with_division(Client):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        Client.__init__(self, id_, client_data, global_data, global_test_data, local_test_data)

    def train(self, mean_pseudo_labels):

        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(self.global_data[self.current_iteration - 1],
                                   batch_size=experiment_config.batch_size, shuffle=False, num_workers=0,
                                   drop_last=True)
        # server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False,
        #                           num_workers=0)
        # print(1)
        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c,
        #                             weight_decay=1e-4)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train_client):
            # print(2)

            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                # print(batch_idx)

                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                # Check for NaN or Inf in outputs

                # Convert model outputs to log probabilities
                outputs_prob = F.log_softmax(outputs, dim=1)
                # Slice pseudo_targets to match the input batch size
                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                # Check if pseudo_targets size matches the input batch size
                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}")
                    continue  # Skip the rest of the loop for this batch

                # Check for NaN or Inf in pseudo targets
                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN or Inf found in pseudo targets at batch {batch_idx}: {pseudo_targets}")
                    continue

                # Normalize pseudo targets to sum to 1
                pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # Calculate the loss
                loss = criterion(outputs_prob, pseudo_targets)

                # Check if the loss is NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train_client}], Loss: {avg_loss:.4f}")

        # self.weights =self.model.state_dict()
        return avg_loss

    def evaluate(self, model=None):
        if model is None:
            model = self.model
        #    print("*** Generating Pseudo-Labels with Probabilities ***")

        # Create a DataLoader for the global data
        global_data_loader = DataLoader(self.global_data[self.current_iteration],
                                        batch_size=experiment_config.batch_size, shuffle=False)

        model.eval()  # Set the model to evaluation mode

        all_probs = []  # List to store the softmax probabilities
        with torch.no_grad():  # Disable gradient computation
            for inputs, _ in global_data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)  # Forward pass

                # Apply softmax to get the class probabilities
                probs = F.softmax(outputs, dim=1)  # Apply softmax along the class dimension

                all_probs.append(probs.cpu())  # Store the probabilities on CPU

        # Concatenate all probabilities into a single tensor (2D matrix)
        all_probs = torch.cat(all_probs, dim=0)

        # print(f"Shape of the 2D pseudo-label matrix: {all_probs.shape}")
        return all_probs




class Server_pFedCK(Server):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict, clients):
        super().__init__(id_, global_data, test_data, clients_ids, clients_test_data_dict)
        self.clients = clients  # List of Client_pFedCK instances

    def calculate_cosine_similarity(self, delta_w1, delta_w2):
        """Flatten and compute cosine similarity."""
        flat1 = torch.cat([v.flatten() for v in delta_w1.values()]).cpu().numpy()
        flat2 = torch.cat([v.flatten() for v in delta_w2.values()]).cpu().numpy()
        dot_product = np.dot(flat1, flat2)
        norm1, norm2 = np.linalg.norm(flat1), np.linalg.norm(flat2)
        return dot_product / (norm1 * norm2 + 1e-8)

    def cluster_clients(self, delta_ws, num_clusters=5):
        n = len(delta_ws)
        similarities = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_cosine_similarity(delta_ws[i], delta_ws[j])
                similarities[i, j] = similarities[j, i] = sim

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(similarities)
        return kmeans.labels_

    def average_parameter_variations(self, delta_ws, cluster_labels):
        cluster_avg = {}
        for cluster_id in np.unique(cluster_labels):
            indices = np.where(cluster_labels == cluster_id)[0]
            avg_variation = copy.deepcopy(delta_ws[indices[0]])
            for key in avg_variation:
                avg_variation[key] = sum(delta_ws[i][key] for i in indices) / len(indices)
            cluster_avg[cluster_id] = avg_variation
        return cluster_avg

    def send_avg_delta_to_clients(self, cluster_avg, cluster_labels):
        for idx, client in enumerate(self.clients):
            cluster_id = cluster_labels[idx]
            avg_delta_w = cluster_avg[cluster_id]
            client.update_interactive_model(avg_delta_w)

    def cluster_and_aggregate(self, delta_ws):
        cluster_labels = self.cluster_clients(delta_ws, num_clusters=5)
        cluster_avg = self.average_parameter_variations(delta_ws, cluster_labels)
        self.send_avg_delta_to_clients(cluster_avg, cluster_labels)


class Server_PseudoLabelsClusters_with_division(Server):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict):
        Server.__init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict)

    def evaluate_for_cluster(self, cluster_id, model=None):
        """
        Evaluate the model using the specified cluster head on the validation data.

        Args:
            cluster_id (int): The ID of the cluster head to evaluate.
            model (nn.Module, optional): The model to evaluate. Defaults to `self.model`.

        Returns:
            torch.Tensor: The concatenated probabilities for the specified cluster head.
        """
        if model is None:
            model = self.model

        print(f"*** Evaluating Cluster {cluster_id} Head ***")
        model.eval()  # Set the model to evaluation mode

        # Use the global validation data for the evaluation
        #

        global_data_loader = DataLoader(self.global_data[self.current_iteration],
                                        batch_size=experiment_config.batch_size, shuffle=False)

        # List to store the probabilities for this cluster
        cluster_probs = []

        with torch.no_grad():  # Disable gradient computation
            for inputs, _ in global_data_loader:
                inputs = inputs.to(device)

                # Evaluate the model using the specified cluster head
                if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
                    outputs = model(inputs, cluster_id=cluster_id)
                if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                    outputs = model(inputs, cluster_id=0)

                # Apply softmax to get class probabilities
                probs = F.softmax(outputs, dim=1)

                # Store probabilities for this cluster head
                cluster_probs.append(probs.cpu())

        # Concatenate all probabilities into a single tensor
        cluster_probs = torch.cat(cluster_probs, dim=0)

        return cluster_probs

    def train(self, mean_pseudo_labels, cluster_num="0", selected_model=None):

        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train *** Cluster: {cluster_num} ***")
        server_loader = DataLoader(self.global_data[self.current_iteration], batch_size=experiment_config.batch_size,
                                   shuffle=False,
                                   num_workers=0, drop_last=True)

        if selected_model is None:
            selected_model_train = self.model
        else:
            selected_model_train = selected_model

        selected_model_train.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(selected_model_train.parameters(), lr=experiment_config.learning_rate_train_s)
        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train_server):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                # Pass `cluster_id` to the model
                outputs = selected_model_train(inputs, cluster_id=cluster_num)

                # Convert model outputs to log probabilities
                outputs_prob = F.log_softmax(outputs, dim=1)

                # Slice pseudo_targets to match the input batch size
                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                # Check if pseudo_targets size matches the input batch size
                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}"
                    )
                    continue

                # Normalize pseudo targets to sum to 1
                pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # Calculate the loss
                loss = criterion(outputs_prob, pseudo_targets)

                # Skip batch if the loss is NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(selected_model_train.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train_server}], Loss: {avg_loss:.4f}")

        return avg_loss


class Server_PseudoLabelsNoServerModel(Server):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict):
        Server.__init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict)

    def iteration_context(self, t):
        self.current_iteration = t
        mean_pseudo_labels_per_cluster_dict, self.clusters_client_id_dict_per_iter[
            t] = self.get_pseudo_labels_input_per_cluster(t)  # #

        for cluster_id, pseudo_labels_for_cluster in mean_pseudo_labels_per_cluster_dict.items():
            for client_id in self.clusters_client_id_dict_per_iter[t][cluster_id]:
                self.pseudo_label_to_send[client_id] = pseudo_labels_for_cluster


class Server_Centralized(Server):
    def __init__(self, id_, train_data, test_data, evaluate_every):
        LearningEntity.__init__(self, id_, None, None)

        self.train_data = self.break_the_dict_structure(train_data)
        self.test_data = self.break_the_dict_structure(test_data)

        self.evaluate_every = evaluate_every

        self.num = (1000) * 17

        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
            raise Exception("todo")

        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            self.multi_model_dict = {}

            self.accuracy_per_cluster_model = {}

            for cluster_id in self.train_data.keys():
                self.multi_model_dict[cluster_id] = get_server_model()
                self.multi_model_dict[cluster_id].apply(self.initialize_weights)
                self.accuracy_per_cluster_model[cluster_id] = {}

    def iteration_context(self, t):
        for cluster_id, model in self.multi_model_dict.items():
            self.fine_tune(model, cluster_id, self.train_data[cluster_id], self.test_data[cluster_id])

    def fine_tune(self, model, cluster_id, train, test):
        print("*** " + self.__str__() + " fine-tune ***")

        fine_tune_loader = DataLoader(train, batch_size=experiment_config.batch_size, shuffle=True)
        model.train()  # Set the model to training mode

        # Define loss function and optimizer

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # experiment_config.learning_rate_train_s)

        epochs = experiment_config.epochs_num_input_fine_tune_centralized_server
        for epoch in range(epochs):
            self.epoch_count += 1
            epoch_loss = 0
            for inputs, targets in fine_tune_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % self.evaluate_every == 0 and epoch != 0:
                self.accuracy_per_cluster_model[cluster_id][epoch] = self.evaluate_accuracy(test, model=model, k=1)

            result_to_print = epoch_loss / len(fine_tune_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {result_to_print:.4f}")
        # self.weights = self.model.state_dict()self.weights = self.model.state_dict()

    def break_the_dict_structure(self, data_):
        ans = {}
        if isinstance(experiment_config.num_clusters, int):
            if experiment_config.num_clusters != 1:
                raise Exception("program only 1 cluster case")

            images = []
            for group_name, torches_list in data_.items():
                for single_torch in torches_list:
                    for image in single_torch:
                        images.append(image)
            ans["group_1.0"] = transform_to_TensorDataset(images)

        else:
            if experiment_config.num_clusters != "Optimal":
                raise Exception("program only Optimal case")
            for group_name, torches_list in data_.items():
                images_per_torch_list = []
                for single_torch in torches_list:
                    for image in single_torch:
                        images_per_torch_list.append(image)
                ans[group_name] = transform_to_TensorDataset(images_per_torch_list)

        return ans


class ServerFedAvg(Server):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict):
        Server.__init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict)
        self.received_weights = {}
        self.weights_to_send = {}

    def iteration_context(self, t):
        self.current_iteration = t
        weights_per_cluster, self.clusters_client_id_dict_per_iter[t] = self.get_weights_per_cluster(
            t)  # #
        for cluster_id, clients_ids_list in self.clusters_client_id_dict_per_iter[t].items():
            for client_id in clients_ids_list:
                self.weights_to_send[client_id] = weights_per_cluster[cluster_id]

    def get_weights_per_cluster(self, t):
        mean_per_cluster = {}

        if experiment_config.num_clusters == "Optimal":
            clusters_client_id_dict = experiment_config.known_clusters

        elif experiment_config.num_clusters == 1:
            clusters_client_id_dict = {0: []}
            for client_id in self.clients_ids:
                clusters_client_id_dict[0].append(client_id)

        else:
            raise Exception("implemented 1 and optimal only")

        cluster_weights_dict = self.get_cluster_weights_dict(clusters_client_id_dict)

        # if experiment_config.num_clusters>1:
        for cluster_id, weights in cluster_weights_dict.items():
            mean_per_cluster[cluster_id] = self.average_weights(weights)
        return mean_per_cluster, clusters_client_id_dict

    def average_weights(self, weights_list):
        """
        Averages a list of state_dicts (model weights) using Federated Averaging (FedAvg).

        :param weights_list: List of model state_dicts
        :return: Averaged state_dict
        """
        if not weights_list:
            raise ValueError("The weights list is empty")

        # Initialize an empty dictionary to store averaged weights
        avg_weights = {}

        # Iterate through each parameter key in the model
        for key in weights_list[0].keys():
            # Stack all weights along a new dimension and take the mean
            avg_weights[key] = torch.stack([weights[key] for weights in weights_list]).mean(dim=0)

        return avg_weights

    def get_cluster_weights_dict(self, clusters_client_id_dict):
        ans = {}
        for cluster_id, clients_ids in clusters_client_id_dict.items():
            ans[cluster_id] = []
            for client_id in clients_ids:
                ans[cluster_id].append(self.received_weights[client_id])
        return ans

class Client_SCAFFOLD(Client):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        Client.__init__(self, id_, client_data, global_data, global_test_data, local_test_data)
        self.weights_to_send = None
        self.weights_received = None
        # Initialize local control variate c_i
        self.c_local = None
        self.c_diff = None
        self.initial_weights = None

    def iteration_context(self, t):
        self.current_iteration = t
        
        if t == 0:
            # Initialize local control variate c_i with zeros
            self.c_local = [torch.zeros_like(param) for param in self.model.parameters()]
            self.model.apply(self.initialize_weights)
            self.initial_weights = [param.clone().detach() for param in self.model.parameters()]
        else:
            # Load received weights
            self.model.load_state_dict(self.weights_received)
            # Store initial weights for this iteration
            self.initial_weights = [param.clone().detach() for param in self.model.parameters()]
        
        # Always train with SCAFFOLD (in first iteration, c_diff will be zero)
        self.train_scaffold()
        
        # Calculate weight differences for server aggregation
        self.calculate_weight_differences()
        
        # Send weights to server
        self.weights_to_send = self.model.state_dict()
        
        # Evaluate accuracy
        self.accuracy_per_client_1[t] = self.evaluate_accuracy_single(self.local_test_set, k=1)
        self.accuracy_per_client_5[t] = self.evaluate_accuracy(self.local_test_set, k=5)
        self.accuracy_per_client_10[t] = self.evaluate_accuracy(self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)

    def calculate_weight_differences(self):
        """Calculate the difference between current and initial weights for SCAFFOLD."""
        if self.initial_weights is not None:
            self.weight_differences = []
            for current_param, initial_param in zip(self.model.parameters(), self.initial_weights):
                self.weight_differences.append(current_param - initial_param)

    def train_scaffold(self):
        """Train with SCAFFOLD correction terms"""
        print("*** " + self.__str__() + " train with SCAFFOLD ***")
        
        train_loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size, shuffle=True)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c)
        
        epochs = experiment_config.epochs_num_train_client
        for epoch in range(epochs):
            self.epoch_count += 1
            epoch_loss = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Add SCAFFOLD correction terms to gradients
                loss.backward()
                if self.c_diff is not None:
                    for param, c_diff in zip(self.model.parameters(), self.c_diff):
                        if param.grad is not None:
                            param.grad += c_diff.data
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        
        return avg_loss

    def set_control_variates(self, c_global, c_local):
        """Set the control variates for SCAFFOLD training."""
        self.c_diff = [c_g - c_l for c_g, c_l in zip(c_global, c_local)]

    def get_control_variates(self):
        """Get the local control variate and weight differences for server aggregation."""
        return self.c_local, self.weight_differences if hasattr(self, 'weight_differences') else None


class Server_SCAFFOLD(Server):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict):
        Server.__init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict)
        self.received_weights = {}
        self.weights_to_send = {}
        self.c_global = None  # Global control variate
        self.c_local_dict = {}  # Dictionary to store local control variates for each client
        self.clients = None  # Will be set when clients are created
        self.accuracy_scaffold_server_1 = {}
        self.accuracy_scaffold_server_5 = {}
        self.accuracy_scaffold_server_10 = {}
        self.accuracy_scaffold_server_100 = {}

        # SCAFFOLD needs a server model for aggregation, even with no_model technique
        if self.model is None:
            self.model = get_server_model()
            self.model.apply(self.initialize_weights)

    def set_clients(self, clients):
        """Set the clients list for control variate management."""
        self.clients = clients

    def iteration_context(self, t):
        self.current_iteration = t
        
        if t == 0:
            # Initialize global control variate c_global with zeros
            self.c_global = [torch.zeros_like(param) for param in self.model.parameters()]
        
        # Collect weights and control variates from clients
        weights_per_cluster, self.clusters_client_id_dict_per_iter[t] = self.get_weights_per_cluster(t)
        
        # Update global model and control variates
        self.aggregate_scaffold(weights_per_cluster)
        
        # Update control variates for next iteration
        self.update_control_variates()
        
        # Evaluate Accuracy
        self.accuracy_scaffold_server_1[t] = self.evaluate_accuracy_single(self.test_global_data, k=1)
        self.accuracy_scaffold_server_5[t] = self.evaluate_accuracy(self.test_global_data, k=5)
        self.accuracy_scaffold_server_10[t] = self.evaluate_accuracy(self.test_global_data, k=10)
        self.accuracy_scaffold_server_100[t] = self.evaluate_accuracy(self.test_global_data, k=100)

        # Ensure all clients have control variates in c_local_dict (initialize with zeros if missing)
        for client_id in self.clients_ids:
            if client_id not in self.c_local_dict:
                # Initialize with zeros for clients that don't have control variates yet
                self.c_local_dict[client_id] = [torch.zeros_like(param) for param in self.model.parameters()]
        
        # Send updated weights and control variates to clients
        for cluster_id, clients_ids_list in self.clusters_client_id_dict_per_iter[t].items():
            for client_id in clients_ids_list:
                self.weights_to_send[client_id] = weights_per_cluster[cluster_id]
                # Also send control variates to clients
                if client_id in self.c_local_dict:
                    client = next((c for c in self.clients if c.id_ == client_id), None)
                    if client is not None:
                        client.set_control_variates(self.c_global, self.c_local_dict[client_id])

    def get_weights_per_cluster(self, t):
        mean_per_cluster = {}

        if experiment_config.num_clusters == "Optimal":
            clusters_client_id_dict = experiment_config.known_clusters
        elif experiment_config.num_clusters == 1:
            clusters_client_id_dict = {0: []}
            for client_id in self.clients_ids:
                clusters_client_id_dict[0].append(client_id)
        else:
            raise Exception("implemented 1 and optimal only")

        cluster_weights_dict = self.get_cluster_weights_dict(clusters_client_id_dict)

        for cluster_id, weights in cluster_weights_dict.items():
            mean_per_cluster[cluster_id] = self.average_weights(weights)
        
        return mean_per_cluster, clusters_client_id_dict

    def get_cluster_weights_dict(self, clusters_client_id_dict):
        ans = {}
        for cluster_id, clients_ids in clusters_client_id_dict.items():
            ans[cluster_id] = []
            for client_id in clients_ids:
                ans[cluster_id].append(self.received_weights[client_id])
        return ans

    def average_weights(self, weights_list):
        """Averages a list of state_dicts (model weights) using Federated Averaging (FedAvg)."""
        if not weights_list:
            raise ValueError("The weights list is empty")

        avg_weights = {}
        for key in weights_list[0].keys():
            avg_weights[key] = torch.stack([weights[key] for weights in weights_list]).mean(dim=0)
        return avg_weights

    def aggregate_scaffold(self, weights_per_cluster):
        """Aggregate weights and update global control variate using SCAFFOLD algorithm."""
        # Update global model with averaged weights
        for cluster_id, avg_weights in weights_per_cluster.items():
            # Update global model parameters
            for name, param in self.model.named_parameters():
                if name in avg_weights:
                    param.data = avg_weights[name].data.clone()

    def update_control_variates(self):
        """Update global control variate based on client updates."""
        if not self.clients:
            return
            
        # Collect control variates from all clients
        c_delta_cache = []
        for client in self.clients:
            c_local, weight_diffs = client.get_control_variates()
            if c_local is not None and weight_diffs is not None:
                # Calculate c_delta for this client
                c_delta = []
                for c_l, w_diff in zip(c_local, weight_diffs):
                    # c_delta = c_local - (1/(K*eta)) * weight_difference
                    # where K is number of local epochs, eta is learning rate
                    coef = 1.0 / (experiment_config.epochs_num_train_client * experiment_config.learning_rate_train_c)
                    c_delta.append(c_l - coef * w_diff)
                c_delta_cache.append(c_delta)
        
        # Update global control variate
        if c_delta_cache:
            for i, c_g in enumerate(self.c_global):
                c_delta_avg = torch.stack([c_delta[i] for c_delta in c_delta_cache]).mean(dim=0)
                c_g.data += c_delta_avg

    def set_client_control_variates(self, client_id, c_local):
        """Set the local control variate for a specific client."""
        self.c_local_dict[client_id] = c_local

    def get_client_control_variates(self, client_id):
        """Get the local control variate for a specific client."""
        return self.c_local_dict.get(client_id, None)