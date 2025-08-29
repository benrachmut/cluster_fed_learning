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
        # self.weights = None
        self.accuracy_per_client_1 = {}
        self.accuracy_per_client_10 = {}
        self.accuracy_per_client_100 = {}
        self.accuracy_per_client_5 = {}
        self.size_sent = {}
        # self.accuracy_per_client_5 = {}

        # self.accuracy_pl_measures= {}
        # self.accuracy_test_measures_k_half_cluster = {}
        # self.accuracy_pl_measures_k_half_cluster= {}

    def initialize_weights(self, layer):
        """Initialize weights for the model layers."""
        # Ensure the seed is within the valid range for PyTorch
        safe_seed = int(self.seed) % (2 ** 31)

        # Update the seed with experiment_config.seed_num
        self.seed = (self.seed + 1)
        torch.manual_seed(self.seed)  # For PyTorch
        torch.cuda.manual_seed(self.seed)  # For CUDA (if using GPU)
        torch.cuda.manual_seed_all(self.seed)  # For multi-GPU

        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # def set_weights(self):
    # if self.weights  is None:
    #    self.model.apply(self.initialize_weights)
    # else:
    #    self.model.apply(self.weights)

    def get_pseudo_label_L2(self, pseudo_labels):
        loader = DataLoader(self.global_data, batch_size=len(self.global_data))
        X_tensor, Y_tensor = next(iter(loader))  # Gets all data
        # Convert to NumPy (if needed)
        ground_truth = Y_tensor.numpy()

        num_classes = pseudo_labels.shape[1]
        ground_truth_onehot = F.one_hot(torch.tensor(ground_truth), num_classes=num_classes).float().numpy()

        return np.mean(np.linalg.norm(pseudo_labels - ground_truth_onehot, axis=1) ** 2)

    def iterate(self, t):
        # self.set_weights()
        torch.manual_seed(self.num + t * 17)
        torch.cuda.manual_seed(self.num + t * 17)
        if experiment_config.is_with_memory_load:
            if t == 0 and self.id_ != "server":
                self.iteration_context(t)
            elif t > 0 and self.id_ != "server":
                # Load the model from file
                self.model = get_client_model(self.rnd_net)
                self.model.load_state_dict(torch.load("./models/model_{}.pth".format(self.id_)))
                self.iteration_context(t)
            elif self.id_ == "server":
                self.iteration_context(t)

            if self.id_ != "server":
                # Save the model state after training to a file
                torch.save(self.model.state_dict(), "./models/model_{}.pth".format(self.id_))
                # Unload the model from memory
                del self.model
        else: self.iteration_context(t)


    @abstractmethod
    def iteration_context(self, t):
        pass

    def evaluate(self, model=None):
        if model is None:
            model = self.model
        #    print("*** Generating Pseudo-Labels with Probabilities ***")

        # Create a DataLoader for the global data
        global_data_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False)

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

    def evaluate_max_accuracy_per_point(self, models, data_, k=1, cluster_id=None):
        """
        Evaluate the per-point max accuracy across multiple models.

        Args:
            models (List[torch.nn.Module]): List of models to evaluate.
            data_ (torch.utils.data.Dataset): The dataset to evaluate.
            k (int): Top-k accuracy. Currently supports only top-1.
            cluster_id (int or None): The cluster ID for multi-head models.

        Returns:
            float: Average of maximum accuracy per data point across all models.
        """
        assert k == 1, "Only top-1 accuracy is currently supported."
        test_loader = DataLoader(data_, batch_size=1, shuffle=False)
        total_points = len(data_)

        # Initialize per-point correct predictions (rows: models, cols: data points)
        model_correct_matrix = torch.zeros((len(models), total_points), dtype=torch.bool)

        for model_idx, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs, cluster_id=cluster_id)
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(0)  # make it [1, num_classes]
                    preds = outputs.argmax(dim=1)
                    model_correct_matrix[model_idx, i] = (preds == targets).item()

        # For each data point, check if any model got it correct
        max_correct_per_point = model_correct_matrix.any(dim=0).float()

        # Compute average over all data points
        average_max_accuracy = max_correct_per_point.mean().item() * 100  # percent

        print(f"Average max accuracy across models: {average_max_accuracy:.2f}%")
        return average_max_accuracy

    def evaluate_accuracy_single(self, data_, model=None, k=1, cluster_id=None):
        if model is None:
            model = self.model
        model.eval()  # Set the model to evaluation mode
        correct = 0  # To count the correct predictions
        total = 0  # To count the total predictions

        test_loader = DataLoader(data_, batch_size=experiment_config.batch_size, shuffle=False)

        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass through the specific cluster head
                outputs = model(inputs, cluster_id=cluster_id)

                # Get the top-1 predictions directly
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)

                top_1_preds = outputs.argmax(dim=1)

                # Update the total number of predictions and correct predictions
                total += targets.size(0)
                correct += (top_1_preds == targets).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0.0
        print(f"Accuracy for cluster {cluster_id if cluster_id is not None else 'default'}: {accuracy:.2f}%")
        return accuracy

    def evaluate_accuracy(self, data_, model=None, k=1, cluster_id=None):
        if model is None:
            model = self.model

        """
        Evaluate the top-k accuracy of the model on the given dataset.

        Args:
            data_ (torch.utils.data.Dataset): The dataset to evaluate.
            model (torch.nn.Module): The model to evaluate.
            k (int): Top-k accuracy.
            cluster_id (int or None): Used if model has multiple heads.

        Returns:
            float: Top-k accuracy (%) on the dataset.
        """
        model.eval()
        correct = 0
        total = 0

        test_loader = DataLoader(data_, batch_size=experiment_config.batch_size, shuffle=False)

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs, cluster_id=cluster_id)

                # Ensure outputs has correct dimensions
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)  # make it [1, num_classes]

                # Top-k predictions (returns both values and indices)
                if experiment_config.num_classes < k:
                    return 0
                else:
                    _, topk_preds = outputs.topk(k, dim=1)
                # Check if the correct label is in the top-k predictions
                correct += (topk_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
                total += targets.size(0)

        accuracy = 100 * correct / total if total > 0 else 0.0
        print(f"Top-{k} Accuracy for cluster {cluster_id if cluster_id is not None else 'default'}: {accuracy:.2f}%")
        return accuracy

    def evaluate_test_loss(self, cluster_id=None, model=None):
        """
        Evaluate the model on the test set and return the loss for a specific cluster head.

        Args:
            cluster_id (int, optional): The cluster head to use. Defaults to None, which evaluates the single head.
            model (nn.Module, optional): The model to evaluate. Defaults to `self.model`.

        Returns:
            float: The average test loss for the specified cluster head.
        """
        if model is None:
            model = self.model

        model.eval()  # Set the model to evaluation mode
        test_loader = DataLoader(self.test_set, batch_size=experiment_config.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()  # Define the loss function

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to device

                # Get the model outputs for the specified cluster head
                outputs = model(inputs, cluster_id=cluster_id)

                # Calculate the loss
                loss = criterion(outputs, targets)

                # Accumulate the loss
                total_loss += loss.item() * inputs.size(0)  # Multiply by the batch size
                total_samples += inputs.size(0)  # Count the number of samples

        # Calculate the average test loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        print(f"Iteration [{self.current_iteration}], Test Loss (Cluster {cluster_id}): {avg_loss:.4f}")

        return avg_loss  # Return the average loss


class Client(LearningEntity):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        LearningEntity.__init__(self, id_, global_data, global_test_data)
        self.num = (self.id_ + 1) * 17
        self.local_test_set = local_test_data
        self.rnd_net = Random((self.seed+1)*17+13+(id_+1)*17 )

        self.local_data = client_data
        self.epoch_count = 0
        self.model = get_client_model(self.rnd_net)
        self.model.apply(self.initialize_weights)
        # self.train_learning_rate = experiment_config.learning_rate_train_c
        # self.weights = None
        self.global_data = global_data
        self.server = None
        self.pseudo_label_L2 = {}
        self.global_label_distribution = self.get_label_distribution()

    def get_label_distribution(self):
        label_counts = defaultdict(int)

        for _, label in self.local_data:
            label_counts[label.item() if hasattr(label, 'item') else int(label)] += 1

        return dict(label_counts)

    def train_with_consistency_and_weights(self, pseudo_label_received):
        print(f"Mean pseudo-labels shape: {pseudo_label_received.shape}")
        print(f"*** {self.__str__()} train ***")

        server_loader = DataLoader(
            self.global_data,
            batch_size=experiment_config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        self.model.train()
        lambda_consistency = experiment_config.lambda_consistency  # Can tune this
        criterion_consistency = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c)
        pseudo_targets_all = pseudo_label_received.to(device)

        # Simple perturbation function (add Gaussian noise)
        def add_noise(inputs, std=0.05):
            noise = torch.randn_like(inputs) * std
            return torch.clamp(inputs + noise, 0., 1.)

        for epoch in range(experiment_config.epochs_num_train_client):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, true_labels) in enumerate(server_loader):
                inputs = inputs.to(device)
                true_labels = true_labels.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                outputs_log_prob = F.log_softmax(outputs, dim=1)

                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                # ...
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(f"Skipping batch {batch_idx}: Pseudo target size mismatch.")
                    continue
                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN/Inf in pseudo targets at batch {batch_idx}")
                    continue

                # REPLACED softmax -> clamp (targets are already probs)
                eps = 1e-8
                pseudo_targets = pseudo_targets.clamp(min=eps, max=1 - eps)
                # ...

                # Compute weights based on global label distribution
                weights = torch.tensor(
                    [self.get_global_label_distribution(label.item()) / len(self.global_data) for label in true_labels],
                    dtype=torch.float32, device=device
                ).unsqueeze(1)  # (batch_size, 1)

                # KL divergence per sample
                loss_kl_per_sample = F.kl_div(outputs_log_prob, pseudo_targets, reduction='none').sum(dim=1)
                loss_kl = (loss_kl_per_sample * weights.squeeze()).mean()

                # Input consistency regularization
                inputs_aug = add_noise(inputs)
                with torch.no_grad():
                    outputs_aug = self.model(inputs_aug)
                    probs = F.softmax(outputs, dim=1)
                    probs_aug = F.softmax(outputs_aug, dim=1)

                loss_consistency = criterion_consistency(probs, probs_aug)

                # Total loss
                loss = loss_kl + lambda_consistency * loss_consistency

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss at batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train_client}], Loss: {avg_loss:.4f}")

        return avg_loss

    def iteration_context(self, t):
        self.current_iteration = t
        for _ in range(10):
            if t > 1:
                if t == 1:
                    self.model.apply(self.initialize_weights)

                if experiment_config.input_consistency == InputConsistency.withInputConsistency:
                    if experiment_config.weights_for_ps:
                        train_loss = self.train_with_consistency_and_weights(self.pseudo_label_received)
                    else:
                        train_loss = self.train_with_consistency(self.pseudo_label_received)
                else:
                    if experiment_config.weights_for_ps:
                        train_loss = self.train_with_weights(self.pseudo_label_received)

                    else:
                        train_loss = self.train(self.pseudo_label_received)

            if t == 0:
                train_loss = self.fine_tune(50)
            else:
                train_loss = self.fine_tune()

            self.pseudo_label_to_send = self.evaluate()
            what_to_send = self.pseudo_label_to_send

            self.size_sent[t] = (what_to_send.numel() * what_to_send.element_size()) / (1024 * 1024)
            self.pseudo_label_L2[t] = self.get_pseudo_label_L2(what_to_send)
            acc = self.evaluate_accuracy_single(self.local_test_set)

            acc_test = self.evaluate_accuracy_single(self.test_global_data)
            if experiment_config.data_set_selected == DataSet.CIFAR100:
                if acc != 1 and acc_test != 1:
                    break
                else:
                    self.model.apply(self.initialize_weights)
            if experiment_config.data_set_selected == DataSet.CIFAR10 or experiment_config.data_set_selected == DataSet.SVHN:
                if acc != 10 and acc_test != 10:
                    break
                else:
                    self.model.apply(self.initialize_weights)
            if experiment_config.data_set_selected == DataSet.TinyImageNet:
                if acc != 0.5 and acc_test != 0.5:
                    break
                else:
                    self.model.apply(self.initialize_weights)

            if experiment_config.data_set_selected == DataSet.EMNIST_balanced:
                if acc > 2.14 and acc_test > 2.14:
                    break
                else:
                    self.model.apply(self.initialize_weights)
        print("hi")

        # self.print_grad_size()

        self.accuracy_per_client_1[t] = self.evaluate_accuracy_single(self.local_test_set, k=1)
        self.accuracy_per_client_10[t] = self.evaluate_accuracy(self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)

        self.accuracy_per_client_5[t] = self.evaluate_accuracy(self.local_test_set, k=5)

    def train__(self, mean_pseudo_labels, data_):

        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(data_, batch_size=experiment_config.batch_size, shuffle=False, num_workers=0,
                                   drop_last=True)

        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train_client):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
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

        self.weights = self.model.state_dict()
        return avg_loss

    def train_with_consistency(self, mean_pseudo_labels):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train ***")

        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False,
                                   num_workers=0, drop_last=True)
        self.model.train()

        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        lambda_consistency = experiment_config.lambda_consistency  # You can tune this
        criterion_consistency = nn.MSELoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c)
        pseudo_targets_all = mean_pseudo_labels.to(device)

        # Simple perturbation function (add Gaussian noise)
        def add_noise(inputs, std=0.05):
            noise = torch.randn_like(inputs) * std
            return torch.clamp(inputs + noise, 0., 1.)

        for epoch in range(experiment_config.epochs_num_train_client):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                outputs_log_prob = F.log_softmax(outputs, dim=1)

                # Index pseudo targets
                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(f"Skipping batch {batch_idx}: Pseudo target size mismatch.")
                    continue
                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN/Inf in pseudo targets at batch {batch_idx}")
                    continue

                pseudo_targets = F.softmax(pseudo_targets, dim=1)
                loss_kl = criterion_kl(outputs_log_prob, pseudo_targets)

                # Input consistency regularization
                inputs_aug = add_noise(inputs)
                with torch.no_grad():
                    outputs_aug = self.model(inputs_aug)
                    probs = F.softmax(outputs, dim=1)
                    probs_aug = F.softmax(outputs_aug, dim=1)
                loss_consistency = criterion_consistency(probs, probs_aug)

                # Total loss
                loss = loss_kl + lambda_consistency * loss_consistency

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss at batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train_client}], Loss: {avg_loss:.4f}")

        return avg_loss

    def get_global_label_distribution(self, k):
        return self.global_label_distribution.get(k, 0)

    def train_with_weights(self, mean_pseudo_labels):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train ***")

        server_loader = DataLoader(
            self.global_data,
            batch_size=experiment_config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c)
        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train_client):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, true_labels) in enumerate(server_loader):
                inputs = inputs.to(device)
                true_labels = true_labels.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                outputs_prob = F.log_softmax(outputs, dim=1)

                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}")
                    continue

                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN or Inf found in pseudo targets at batch {batch_idx}: {pseudo_targets}")
                    continue

                pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # Get weights based on true labels
                weights = torch.tensor(
                    [self.get_global_label_distribution(label.item()) / len(self.global_data) for label in true_labels],
                    dtype=torch.float32, device=device
                ).unsqueeze(1)  # (batch_size, 1)

                # KL divergence per sample
                loss_per_sample = F.kl_div(outputs_prob, pseudo_targets, reduction='none').sum(dim=1)
                weighted_loss = (loss_per_sample * weights.squeeze()).mean()
                loss = weighted_loss

                # print("Type of loss:", type(loss))
                # print("Loss value:", loss)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train_client}], Loss: {avg_loss:.4f}")

        return avg_loss

    def train(self, mean_pseudo_labels):

        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False,
                                   num_workers=0,
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

    def __str__(self):
        return "Client " + str(self.id_)

    def fine_tune(self, num_of_epochs =experiment_config.epochs_num_input_fine_tune_clients):
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

        epochs = num_of_epochs
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
        # self.weights = self.model.state_dict()self.weights = self.model.state_dict()

        return result_to_print

    def fine_tune_with_consistency(self):
        print("*** " + self.__str__() + " fine-tune ***")

        fine_tune_loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size, shuffle=True)
        self.model.train()

        criterion_ce = nn.CrossEntropyLoss()
        criterion_consistency = nn.MSELoss()
        lambda_consistency = experiment_config.lambda_consistency  # You can tune this value

        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_fine_tune_c)

        def add_noise(inputs, std=0.05):
            noise = torch.randn_like(inputs) * std
            return torch.clamp(inputs + noise, 0., 1.)

        epochs = experiment_config.epochs_num_input_fine_tune_clients
        for epoch in range(epochs):
            self.epoch_count += 1
            epoch_loss = 0

            for inputs, targets in fine_tune_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss_ce = criterion_ce(outputs, targets)

                # Input consistency regularization
                inputs_aug = add_noise(inputs)
                with torch.no_grad():
                    outputs_aug = self.model(inputs_aug)
                    probs = F.softmax(outputs, dim=1)
                    probs_aug = F.softmax(outputs_aug, dim=1)

                loss_consistency = criterion_consistency(probs, probs_aug)

                # Total loss
                loss = loss_ce + lambda_consistency * loss_consistency

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            result_to_print = epoch_loss / len(fine_tune_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {result_to_print:.4f}")

        return result_to_print

    def print_grad_size(self):

        # Measure total gradient size (L2 norm and memory in bytes)
        total_grad_elements = 0
        total_grad_bytes = 0
        grad_norms = []

        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_elements += param.grad.numel()
                total_grad_bytes += param.grad.numel() * param.grad.element_size()
                grad_norms.append(param.grad.detach().norm(2))

        if grad_norms:
            total_l2_norm = torch.norm(torch.stack(grad_norms), 2).item()
            print(f"Total gradient L2 norm: {total_l2_norm:.4f}")
        print(f"Total gradient elements: {total_grad_elements}")
        print(f"Total gradient size: {total_grad_bytes / 1024 / 1024:.4f} MB")
        print()


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


class Server(LearningEntity):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict):
        super().__init__(id_, global_data, test_data)

        self.num = (1000) * 17
        self.clients_ids = clients_ids
        self.clients_test_data_dict = clients_test_data_dict

        self.pseudo_label_received = {}
        self.pseudo_label_to_send = {}
        self.clusters_client_id_dict_per_iter = {}
        self.reset_clients_received_pl()

        # metrics stores
        self.pseudo_label_before_net_L2 = {}
        self.pseudo_label_after_net_L2 = {}
        self.accuracy_server_test_1 = {}
        self.accuracy_global_data_1 = {}
        self.accuracy_test_max = {}
        self.accuracy_global_max = {}

        self.accuracy_per_client_1_max = {}
        self.accuracy_per_client_10_max = {}
        self.accuracy_per_client_100_max = {}
        self.accuracy_per_client_5_max = {}

        for client_id in self.clients_ids:
            self.accuracy_per_client_1[client_id] = {}
            self.accuracy_per_client_1_max[client_id] = {}
            self.accuracy_per_client_5_max[client_id] = {}
            self.accuracy_per_client_10_max[client_id] = {}
            self.accuracy_per_client_100_max[client_id] = {}

        # cluster bookkeeping
        if isinstance(experiment_config.num_clusters, int):
            num_clusters = experiment_config.num_clusters
        else:
            num_clusters = experiment_config.number_of_optimal_clusters

        self.previous_centroids_dict = {}
        if num_clusters > 0:
            for cluster_id in range(num_clusters):
                self.previous_centroids_dict[cluster_id] = None
                self.accuracy_server_test_1[cluster_id] = {}
                self.accuracy_global_data_1[cluster_id] = {}

        # models
        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
            self.model = get_server_model()
            self.model.apply(self.initialize_weights)

        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            self.multi_model_dict = {}
            for cluster_id in range(num_clusters):
                m = get_server_model()
                m.apply(self.initialize_weights)
                self.multi_model_dict[cluster_id] = m

        # ---------- NEW: server KD knobs & persistent optimizers ----------
        self.kd_temperature = getattr(experiment_config, "kd_temperature", 2.0)
        self.pl_conf_thresh = getattr(experiment_config, "pl_conf_thresh", 0.6)

        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            self.server_opt = {
                cid: torch.optim.Adam(
                    self.multi_model_dict[cid].parameters(),
                    lr=experiment_config.learning_rate_train_s
                )
                for cid in self.multi_model_dict
            }
        else:
            self.server_opt = None

        # diagnostics storage (per-cluster last pseudo-labels on U)
        self._prev_pl = {}

    # ---------------- basic plumbing ----------------
    def __str__(self):
        return "server"

    def receive_single_pseudo_label(self, sender, info):
        self.pseudo_label_received[sender] = info

    def reset_clients_received_pl(self):
        for id_ in self.clients_ids:
            self.pseudo_label_received[id_] = None

    # ---------------- clustering helpers ----------------
    def centroids_are_empty(self):
        for prev_cent in self.previous_centroids_dict.values():
            if prev_cent is None:
                return True
        else:
            return False

    def k_means_grouping(self):
        k = experiment_config.num_clusters
        client_data = self.pseudo_label_received

        client_ids = list(client_data.keys())
        pseudo_labels = [client_data[client_id].flatten().numpy() for client_id in client_ids]
        data_matrix = np.vstack(pseudo_labels)

        if not self.centroids_are_empty():
            previous_centroids_array = np.array(list(self.previous_centroids_dict.values()))
            if previous_centroids_array.shape[0] != k:
                raise ValueError(f"Previous centroids count does not match k={k}.")
            kmeans = KMeans(n_clusters=k, init=previous_centroids_array, n_init=1, random_state=42)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42)

        kmeans.fit(data_matrix)
        self.previous_centroids_dict = {i: centroid for i, centroid in enumerate(kmeans.cluster_centers_)}

        cluster_assignments = kmeans.predict(data_matrix)
        clusters = {i: [] for i in range(k)}
        for client_id, cluster in zip(client_ids, cluster_assignments):
            clusters[cluster].append(client_id)
        return clusters

    def manual_grouping(self):
        if isinstance(experiment_config.num_clusters, int):
            num_clusters = experiment_config.num_clusters
        else:
            num_clusters = experiment_config.number_of_optimal_clusters

        if num_clusters == 1:
            return {0: self.clients_ids}
        return self.get_clusters_centers_dict()

    # (distance helpers unchanged)
    def calc_L2(self, pair):
        first_pl = self.pseudo_label_received[pair[0]]
        second_pl = self.pseudo_label_received[pair[1]]
        return Server.calc_L2_given_pls(first_pl, second_pl)

    @staticmethod
    def calc_L2_given_pls(pl1, pl2):
        difference = pl1 - pl2
        squared_difference = difference ** 2
        sum_squared = torch.sum(squared_difference)
        return torch.sqrt(sum_squared).item()

    def calc_cross_entropy(self, pair):
        first_pl = self.pseudo_label_received[pair[0]]
        second_pl = self.pseudo_label_received[pair[1]]
        return Server.calc_cross_entropy_given_pl(first_pl, second_pl)

    @staticmethod
    def calc_cross_entropy_given_pl(first_pl, second_pl):
        loss1 = -(first_pl * torch.log(second_pl)).sum(dim=1).mean()
        loss2 = -(second_pl * torch.log(first_pl.clamp(min=1e-9))).sum(dim=1).mean()
        return (0.5 * (loss1 + loss2)).item()

    def get_distance_dict(self):
        pairs = list(combinations(self.clients_ids, 2))
        distance_dict = {}
        for pair in pairs:
            if experiment_config.cluster_technique in (ClusterTechnique.greedy_elimination_L2, ClusterTechnique.manual_L2):
                distance_dict[pair] = self.calc_L2(pair)
            else:
                distance_dict[pair] = self.calc_cross_entropy(pair)
        return distance_dict

    def get_distance_per_client(self, distance_dict):
        ans = {}
        for pair, dist in distance_dict.items():
            a, b = pair
            if a not in ans: ans[a] = {}
            if b not in ans: ans[b] = {}
            ans[a][b] = dist
            ans[b][a] = dist
        return ans

    def filter_far_clients(self, distance_per_client, epsilon):
        for client, dd in distance_per_client.items():
            far = [other for other, d in dd.items() if d > epsilon]
            for other in far:
                del dd[other]

    def greedy_elimination_t0(self, epsilon_, distance_per_client):
        self.filter_far_clients(distance_per_client, epsilon_)
        clusters_client_id_dict = {}
        counter = -1
        while len(distance_per_client) > 0:
            counter += 1
            max_client = max(distance_per_client, key=lambda k: len(distance_per_client[k]))
            others = list(distance_per_client[max_client].keys())
            lst = copy.deepcopy(others); lst.append(max_client)
            clusters_client_id_dict[counter] = lst
            for other in clusters_client_id_dict[counter]:
                for dd in distance_per_client.values():
                    if other in dd: del dd[other]
                del distance_per_client[other]
        return clusters_client_id_dict

    def calc_epsilon(self, distance_per_client):
        epsilon_ = 0.1
        while True:
            amount_clusters = len(self.greedy_elimination_t0(epsilon_, copy.deepcopy(distance_per_client)))
            if amount_clusters <= experiment_config.number_of_optimal_clusters + experiment_config.cluster_addition:
                break
            epsilon_ += 0.1
        with open("results.txt", "a") as f:
            f.write(f"BETA for {experiment_config.cluster_addition} for seed {experiment_config.seed_num} is {epsilon_}\n")
        return epsilon_

    def greedy_elimination(self):
        distance_dict = self.get_distance_dict()
        distance_per_client = self.get_distance_per_client(distance_dict)
        epsilon_ = self.calc_epsilon(distance_per_client)
        clusters_client_id_dict = self.greedy_elimination_t0(epsilon_, distance_per_client)
        experiment_config.num_clusters = len(clusters_client_id_dict)
        self.init_models_measures()
        return clusters_client_id_dict

    def compute_distances(self, id_label_dict, distance_function):
        distance_dict = {}
        ids = list(id_label_dict.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                distance = distance_function(id_label_dict[id1], id_label_dict[id2])
                distance_dict[(id1, id2)] = distance
        return distance_dict

    def get_pseudo_label_in_cluster(self, clusters_client_id_dict):
        pseudo_labels_in_cluster = {}
        for cluster_id, clients in clusters_client_id_dict.items():
            pseudo_labels_in_cluster[cluster_id] = []
            for client_id in clients:
                pseudo_labels_in_cluster[cluster_id].append(self.pseudo_label_received[client_id])
        return pseudo_labels_in_cluster

    def initiate_clusters_centers_dict(self, L2_of_all_clients):
        max_pair = max(L2_of_all_clients.items(), key=lambda item: item[1])[0]
        return {
            max_pair[0]: self.pseudo_label_received[max_pair[0]],
            max_pair[1]: self.pseudo_label_received[max_pair[1]],
        }

    def update_distance_of_all_clients(self, L2_of_all_clients, clusters_centers_dict):
        L2_temp = {}
        for (id1, id2), v in L2_of_all_clients.items():
            if (id1 in clusters_centers_dict) ^ (id2 in clusters_centers_dict):
                L2_temp[(id1, id2)] = v
        return L2_temp

    def get_l2_of_non_centers(self, L2_of_all_clients, clusters_centers_dict):
        all_vals = {}
        for pair, l2 in L2_of_all_clients.items():
            which = pair[1] if pair[0] in clusters_centers_dict else pair[0]
            all_vals.setdefault(which, []).append(l2)
        sums = {k: sum(v) for k, v in all_vals.items()}
        return max(sums, key=sums.get)

    def complete_clusters_centers_and_L2_of_all_clients(self, clusters_centers_dict):
        cluster_counter = (3 if isinstance(experiment_config.num_clusters, str)
                           else experiment_config.num_clusters - 2)
        while cluster_counter > 0:
            distance_all = self.get_distance_dict()
            distance_all = self.update_distance_of_all_clients(distance_all, clusters_centers_dict)
            new_center = self.get_l2_of_non_centers(distance_all, clusters_centers_dict)
            clusters_centers_dict[new_center] = self.pseudo_label_received[new_center]
            cluster_counter -= 1
        distance_all = self.get_distance_dict()
        distance_all = self.update_distance_of_all_clients(distance_all, clusters_centers_dict)
        return distance_all, clusters_centers_dict

    def get_l2_of_non_center_to_center(self, L2_from_center_clients, clusters_centers_dict):
        ans = {}
        for (a, b), l2 in L2_from_center_clients.items():
            if a in clusters_centers_dict:
                not_center, center = b, a
            elif b in clusters_centers_dict:
                not_center, center = a, b
            else:
                continue
            ans.setdefault(not_center, {})[center] = l2
        return ans

    def get_non_center_to_which_center_dict(self, l2_of_non_center_to_center):
        one_to_one = {k: min(v, key=v.get) for k, v in l2_of_non_center_to_center.items()}
        grouped = {}
        for key, value in one_to_one.items():
            grouped.setdefault(value, []).append(key)
        return grouped

    def prep_clusters(self, mapping):
        ans, counter = {}, 0
        for k, lst in mapping.items():
            ans[counter] = [k] + list(lst)
            counter += 1
        return ans

    def center_id_not_in_ans(self, ans, center_id):
        for lst in ans.values():
            if center_id in lst:
                return None
        return center_id

    def get_centers_to_add(self, clusters_centers_dict, ans):
        centers_to_add = []
        for center_id in clusters_centers_dict.keys():
            v = self.center_id_not_in_ans(ans, center_id)
            if v is not None:
                centers_to_add.append(v)
        return centers_to_add

    def get_clusters_centers_dict(self):
        L2_all = self.get_distance_dict()
        centers = self.initiate_clusters_centers_dict(L2_all)
        L2_from_center, centers = self.complete_clusters_centers_and_L2_of_all_clients(centers)
        L2_noncenter_to_center = self.get_l2_of_non_center_to_center(L2_from_center, centers)
        non_center_map = self.get_non_center_to_which_center_dict(L2_noncenter_to_center)
        ans = self.prep_clusters(non_center_map)
        extra = self.get_centers_to_add(centers, ans)
        if len(extra) > 0:
            start = max(ans.keys()) + 1 if len(ans) > 0 else 0
            for i, cent in enumerate(extra):
                ans[start + i] = [cent]
        return ans

    # ---------------- aggregation of client PLs ----------------
    @staticmethod
    def _logit_mean(pl_list, T=1.0, eps=1e-8):
        """Aggregate probability vectors by averaging logits (numerically robust)."""
        logits = [torch.log(pl.clamp(eps, 1 - eps)) for pl in pl_list]
        avg_logit = torch.stack(logits, dim=0).mean(dim=0) / T
        return F.softmax(avg_logit, dim=1)

    def get_pseudo_labels_input_per_cluster(self, timestamp):
        mean_per_cluster = {}
        flag = False
        clusters_client_id_dict = None

        if experiment_config.num_clusters == "Optimal":
            clusters_client_id_dict = experiment_config.known_clusters
            flag = True
        if experiment_config.num_clusters == 1:
            clusters_client_id_dict = {0: self.clients_ids}
            flag = True

        if (experiment_config.cluster_technique in
            (ClusterTechnique.greedy_elimination_cross_entropy,
             ClusterTechnique.greedy_elimination_L2)) and not flag:
            clusters_client_id_dict = self.greedy_elimination() if timestamp == 0 else self.clusters_client_id_dict_per_iter[0]

        if experiment_config.cluster_technique == ClusterTechnique.kmeans and not flag:
            clusters_client_id_dict = self.k_means_grouping()

        if (experiment_config.cluster_technique in
            (ClusterTechnique.manual_L2, ClusterTechnique.manual_cross_entropy)) and not flag:
            clusters_client_id_dict = self.manual_grouping()

        if experiment_config.cluster_technique == ClusterTechnique.manual_single_iter and not flag:
            clusters_client_id_dict = self.manual_grouping() if timestamp == 0 else self.clusters_client_id_dict_per_iter[0]

        # aggregate PLs per cluster
        cluster_mean_pseudo_labels_dict = self.get_cluster_mean_pseudo_labels_dict(clusters_client_id_dict)
        for cluster_id, pseudo_labels in cluster_mean_pseudo_labels_dict.items():
            pl_list = list(pseudo_labels)
            if experiment_config.server_input_tech == ServerInputTech.mean:
                mean_per_cluster[cluster_id] = self._logit_mean(
                    pl_list, T=getattr(experiment_config, "agg_temperature", 1.0)
                )
            elif experiment_config.server_input_tech == ServerInputTech.max:
                mean_per_cluster[cluster_id] = self.select_confident_pseudo_labels(pl_list)
            else:
                # default to logit-mean
                mean_per_cluster[cluster_id] = self._logit_mean(pl_list)

        return mean_per_cluster, clusters_client_id_dict

    def get_cluster_mean_pseudo_labels_dict(self, clusters_client_id_dict):
        ans = {}
        for cluster_id, clients_ids in clusters_client_id_dict.items():
            ans[cluster_id] = []
            for client_id in clients_ids:
                ans[cluster_id].append(self.pseudo_label_received[client_id])
        return ans

    def select_confident_pseudo_labels(self, cluster_pseudo_labels):
        num_data_points = cluster_pseudo_labels[0].size(0)
        max_confidences = torch.zeros(num_data_points, device=cluster_pseudo_labels[0].device)
        selected_labels = torch.zeros_like(cluster_pseudo_labels[0])
        for pseudo_labels in cluster_pseudo_labels:
            cluster_max_confidences, _ = torch.max(pseudo_labels, dim=1)
            mask = cluster_max_confidences > max_confidences
            max_confidences[mask] = cluster_max_confidences[mask]
            selected_labels[mask] = pseudo_labels[mask]
        return selected_labels

    # ---------------- evaluation helpers ----------------
    def evaluate_for_cluster(self, cluster_id, model=None):
        model = self.model if model is None else model
        print(f"*** Evaluating Cluster {cluster_id} Head ***")
        model.eval()
        global_data_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False)
        cluster_probs = []
        with torch.no_grad():
            for inputs, _ in global_data_loader:
                inputs = inputs.to(device)
                if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
                    outputs = model(inputs, cluster_id=cluster_id)
                else:
                    outputs = model(inputs, cluster_id=0)
                probs = F.softmax(outputs, dim=1)
                cluster_probs.append(probs.cpu())
        return torch.cat(cluster_probs, dim=0)

    # ---------------- per-round orchestration ----------------
    def iteration_context(self, t):
        self.current_iteration = t
        pseudo_labels_per_cluster, self.clusters_client_id_dict_per_iter[t] = self.get_pseudo_labels_input_per_cluster(t)
        self.pseudo_label_before_net_L2[t] = {}

        if t > 0:
            for cluster_id, pl in pseudo_labels_per_cluster.items():
                self.pseudo_label_before_net_L2[t][cluster_id] = self.get_pseudo_label_L2(pl)

            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
                self.create_feed_back_to_clients_multihead(pseudo_labels_per_cluster, t)
            else:
                _ = self.create_feed_back_to_clients_multimodel(pseudo_labels_per_cluster, t)

            self.pseudo_label_after_net_L2[t] = {}
            # (you can compute after-net L2 per-cluster here if desired)

            self.evaluate_results(t)

        self.reset_clients_received_pl()

    def create_feed_back_to_clients_multihead(self, mean_pseudo_labels_per_cluster, t):
        # Minimal change: one pass per round per head
        for cluster_id, mean_pseudo_label_for_cluster in mean_pseudo_labels_per_cluster.items():
            self.train(mean_pseudo_label_for_cluster, int(cluster_id))
        if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_cluster:
            for cluster_id in mean_pseudo_labels_per_cluster.keys():
                pseudo_labels_for_cluster = self.evaluate_for_cluster(int(cluster_id))
                for client_id in self.clusters_client_id_dict_per_iter[t][cluster_id]:
                    self.pseudo_label_to_send[client_id] = pseudo_labels_for_cluster
        else:
            pseudo_labels_for_cluster_list = []
            for cluster_id in mean_pseudo_labels_per_cluster.keys():
                pseudo_labels_for_cluster_list.append(self.evaluate_for_cluster(int(cluster_id)))
            pseudo_labels_to_send = self.select_confident_pseudo_labels(pseudo_labels_for_cluster_list)
            for client_id in self.clients_ids:
                self.pseudo_label_to_send[client_id] = pseudo_labels_to_send

    def create_feed_back_to_clients_multimodel(self, mean_pseudo_labels_per_cluster, t):
        pl_per_cluster = {}
        for cluster_id, mean_pseudo_label_for_cluster in mean_pseudo_labels_per_cluster.items():
            selected_model = self.multi_model_dict[cluster_id]

            # train once per round on aggregated PLs (no reinit/early break)
            if experiment_config.input_consistency == InputConsistency.withInputConsistency:
                self.train_with_consistency(mean_pseudo_label_for_cluster, 0, selected_model)
            else:
                self.train(mean_pseudo_label_for_cluster, 0, selected_model)

            # produce PLs for feedback
            pseudo_labels_for_cluster = self.evaluate_for_cluster(0, selected_model)
            pl_per_cluster[cluster_id] = pseudo_labels_for_cluster

            # diagnostics: is the teacher changing?
            with torch.no_grad():
                prev = self._prev_pl.get(cluster_id, None)
                if prev is not None:
                    delta = (pseudo_labels_for_cluster - prev).abs().mean().item()
                    print(f"[server] k={cluster_id} mean|ΔPL|={delta:.6f}")
                self._prev_pl[cluster_id] = pseudo_labels_for_cluster.clone()

            if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_cluster:
                for client_id in self.clusters_client_id_dict_per_iter[t][cluster_id]:
                    self.pseudo_label_to_send[client_id] = pseudo_labels_for_cluster

        if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_client:
            pseudo_labels_for_cluster_list = [pl_per_cluster[cid] for cid in mean_pseudo_labels_per_cluster.keys()]
            pseudo_labels_to_send = self.select_confident_pseudo_labels(pseudo_labels_for_cluster_list)
            for client_id in self.clients_ids:
                self.pseudo_label_to_send[client_id] = pseudo_labels_to_send

        return pl_per_cluster

    def evaluate_results(self, t):
        if isinstance(experiment_config.num_clusters, int):
            num_clusters = experiment_config.num_clusters
        else:
            num_clusters = experiment_config.number_of_optimal_clusters

        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            models_list = list(self.multi_model_dict.values())
            self.accuracy_global_data_1[t] = self.evaluate_max_accuracy_per_point(models=models_list,
                                                                                  data_=self.global_data, k=1,
                                                                                  cluster_id=None)
        for cluster_id in range(num_clusters):
            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                selected_model = self.multi_model_dict[cluster_id]; cluster_id_to_examine = 0
            else:
                selected_model = None; cluster_id_to_examine = cluster_id

            self.accuracy_server_test_1[cluster_id][t] = self.evaluate_accuracy_single(
                self.test_global_data, model=selected_model, k=1, cluster_id=cluster_id_to_examine
            )

        for client_id in self.clients_ids:
            test_data_per_clients = self.clients_test_data_dict[client_id]
            cluster_id_for_client = self.get_cluster_of_client(client_id, t)
            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                selected_model = self.multi_model_dict[cluster_id_for_client]
                cid_eval = 0
            else:
                selected_model = None
                cid_eval = cluster_id_for_client

            self.accuracy_per_client_1[client_id][t] = self.evaluate_accuracy_single(
                test_data_per_clients, model=selected_model, k=1, cluster_id=cid_eval
            )

            # also track "best head" for this client
            l1, l2, l3, l4 = [], [], [], []
            for cluster_id in range(num_clusters):
                if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                    l1.append(self.evaluate_accuracy_single(self.clients_test_data_dict[client_id],
                                                            model=self.multi_model_dict[cluster_id], k=1, cluster_id=0))
                    l2.append(self.evaluate_accuracy(self.clients_test_data_dict[client_id],
                                                     model=self.multi_model_dict[cluster_id], k=10, cluster_id=0))
                    l3.append(self.evaluate_accuracy(self.clients_test_data_dict[client_id],
                                                     model=self.multi_model_dict[cluster_id], k=100, cluster_id=0))
                    l4.append(self.evaluate_accuracy(self.clients_test_data_dict[client_id],
                                                     model=self.multi_model_dict[cluster_id], k=5, cluster_id=0))
                else:
                    l1.append(self.evaluate_accuracy(self.clients_test_data_dict[client_id],
                                                     model=selected_model, k=1, cluster_id=cluster_id))
            print("client_id", client_id, "accuracy_per_client_1_max", max(l1))
            self.accuracy_per_client_1_max[client_id][t]   = max(l1)
            self.accuracy_per_client_10_max[client_id][t]  = max(l2) if len(l2) else None
            self.accuracy_per_client_100_max[client_id][t] = max(l3) if len(l3) else None
            self.accuracy_per_client_5_max[client_id][t]   = max(l4) if len(l4) else None

    # ---------------- server training (KD on U) ----------------
    def train_with_consistency(self, mean_pseudo_labels, cluster_num: int = 0, selected_model=None):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train *** Cluster: {cluster_num} ***")

        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size,
                                   shuffle=False, num_workers=0, drop_last=True)

        selected_model_train = self.model if selected_model is None else selected_model
        selected_model_train.train()

        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_consistency = nn.MSELoss()
        lambda_consistency = experiment_config.lambda_consistency
        T = self.kd_temperature

        # reuse persistent optimizer if available
        if self.server_opt is not None:
            opt = self.server_opt[cluster_num]
        else:
            opt = torch.optim.Adam(selected_model_train.parameters(), lr=experiment_config.learning_rate_train_s)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        def add_noise(inputs, std=0.05):
            noise = torch.randn_like(inputs) * std
            return torch.clamp(inputs + noise, 0., 1.)

        for epoch in range(experiment_config.epochs_num_train_server):
            self.epoch_count += 1
            epoch_loss = 0.0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                opt.zero_grad()

                outputs = selected_model_train(inputs, cluster_id=cluster_num)
                outputs_prob = F.log_softmax(outputs / T, dim=1)

                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(f"Skipping batch {batch_idx}: PL size mismatch.")
                    continue

                eps = 1e-8
                pseudo_targets = pseudo_targets.clamp(min=eps, max=1 - eps)

                # confidence filter
                conf, _ = pseudo_targets.max(dim=1)
                mask = conf >= self.pl_conf_thresh
                if mask.sum() == 0:
                    continue

                outputs_prob_m   = outputs_prob[mask]
                pseudo_targets_m = pseudo_targets[mask]

                loss_kl = criterion_kl(outputs_prob_m, pseudo_targets_m) * (T * T)

                # input consistency on masked subset
                inputs_aug = add_noise(inputs[mask])
                with torch.no_grad():
                    outputs_aug = selected_model_train(inputs_aug, cluster_id=cluster_num)
                    probs      = F.softmax(outputs[mask], dim=1)
                    probs_aug  = F.softmax(outputs_aug,   dim=1)
                loss_consistency = criterion_consistency(probs, probs_aug)

                loss = loss_kl + lambda_consistency * loss_consistency

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf at batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(selected_model_train.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(server_loader))
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train_server}], Loss: {avg_loss:.4f}")

        return avg_loss

    def train(self, mean_pseudo_labels, cluster_num: int = 0, selected_model=None):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train *** Cluster: {cluster_num} ***")

        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size,
                                   shuffle=False, num_workers=0, drop_last=True)

        selected_model_train = self.model if selected_model is None else selected_model
        selected_model_train.train()

        criterion = nn.KLDivLoss(reduction='batchmean')
        T = self.kd_temperature

        if self.server_opt is not None:
            opt = self.server_opt[cluster_num]
        else:
            opt = torch.optim.Adam(selected_model_train.parameters(), lr=experiment_config.learning_rate_train_s)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train_server):
            self.epoch_count += 1
            epoch_loss = 0.0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                opt.zero_grad()

                outputs = selected_model_train(inputs, cluster_id=cluster_num)
                outputs_prob = F.log_softmax(outputs / T, dim=1)

                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(f"Skipping batch {batch_idx}: PL size mismatch.")
                    continue

                eps = 1e-8
                pseudo_targets = pseudo_targets.clamp(min=eps, max=1 - eps)

                # confidence filter
                conf, _ = pseudo_targets.max(dim=1)
                mask = conf >= self.pl_conf_thresh
                if mask.sum() == 0:
                    continue

                outputs_prob_m   = outputs_prob[mask]
                pseudo_targets_m = pseudo_targets[mask]

                loss = criterion(outputs_prob_m, pseudo_targets_m) * (T * T)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf at batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(selected_model_train.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(server_loader))
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train_server}], Loss: {avg_loss:.4f}")

        return avg_loss

    # ---------------- misc ----------------
    def get_cluster_of_client(self, client_id, t):
        if experiment_config.cluster_technique == ClusterTechnique.manual_single_iter:
            for cluster_id, clients_id_list in self.clusters_client_id_dict_per_iter[0].items():
                if client_id in clients_id_list:
                    return cluster_id
        else:
            for cluster_id, clients_id_list in self.clusters_client_id_dict_per_iter[t].items():
                if client_id in clients_id_list:
                    return cluster_id
        print()  # fallback/debug

    def init_models_measures(self):
        num_clusters = experiment_config.num_clusters
        for cluster_id in range(num_clusters):
            self.previous_centroids_dict[cluster_id] = None
            self.accuracy_server_test_1[cluster_id] = {}
            self.accuracy_global_data_1[cluster_id] = {}
            self.multi_model_dict[cluster_id] = get_server_model()
            self.multi_model_dict[cluster_id].apply(self.initialize_weights)
        # rebuild persistent optimizers after reinit
        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            self.server_opt = {
                cid: torch.optim.Adam(self.multi_model_dict[cid].parameters(),
                                      lr=experiment_config.learning_rate_train_s)
                for cid in self.multi_model_dict
            }


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