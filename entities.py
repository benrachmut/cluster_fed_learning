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

import os

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


class DenseNetServer(nn.Module):
    def __init__(self, num_classes, num_clusters=1):
        super(DenseNetServer, self).__init__()

        # Load DenseNet121 without pre-trained weights
        self.densenet = models.densenet121(weights=None)

        # Replace the final classifier to output num_classes
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)

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
        # Resize input for DenseNet
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.densenet(x)  # Backbone feature output

        if self.num_clusters == 1:
            return self.head(x)

        if cluster_id is not None:
            return self.heads[f"head_{cluster_id}"](x)

        return {f"head_{i}": head(x) for i, head in self.heads.items()}


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



def get_rnd_net(rnd:Random = None):
    p = rnd.random()
    if p <= 0.25:
        print("ResNet18Server")
        return ResNet18Server(num_classes=experiment_config.num_classes).to(device), NetType.ResNet
    if 0.25 < p <= 0.50:
        print("MobileNetV2Server")
        return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device), NetType.MobileNet
    if 0.50 < p <= 0.75:
        print("SqueezeNetServer")
        return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device), NetType.SqueezeNet
    else:
        print("AlexNet")
        return AlexNet(num_classes=experiment_config.num_classes).to(device), NetType.ALEXNET

def get_rnd_strong_net(rnd:Random = None):
    p = rnd.random()
    if p <= 0.5:
        print("ResNet18Server")
        return ResNet18Server(num_classes=experiment_config.num_classes).to(device), NetType.ResNet
    # if p <= 0.33:
    #     print("MobileNetV2Server")
    #     return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device)
    # if 0.33 < p <= 0.66:
    #     print("SqueezeNetServer")
    #     return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device)
    else:
        print("AlexNet")
        return AlexNet(num_classes=experiment_config.num_classes).to(device),NetType.ALEXNET

def get_rnd_weak_net(rnd:Random = None):
    p = rnd.random()

    if p <= 0.5:
        print("MobileNetV2Server")
        return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device), NetType.MobileNet
    else:
        print("SqueezeNetServer")
        return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device), NetType.SqueezeNet


def get_rnd_net_weak(rnd:Random = None):
    p = rnd.random()
    if p <= 0.5:
        print("MobileNetV2Server")
        return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device), NetType.MobileNet
    else:
        print("SqueezeNetServer")
        return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device), NetType.SqueezeNet





def get_ResNetSqueeze(rand_client):
    p = rand_client.random()
    if p <= 0.5:
        print("Res")
        return ResNet18Server(num_classes=experiment_config.num_classes).to(device), NetType.ResNet
    else:
        print("SqueezeNetServer")
        return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device), NetType.SqueezeNet


def get_ResNetMobile(rand_client):
    p = rand_client.random()
    if p <= 0.5:
        print("Res")
        return ResNet18Server(num_classes=experiment_config.num_classes).to(device),NetType.ResNet
    else:
        print("Mobile")
        return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device), NetType.MobileNet


def get_AlexMobile(rand_client):
    p = rand_client.random()
    if p <= 0.5:
        print("Alex")
        return AlexNet(num_classes=experiment_config.num_classes).to(device),NetType.ALEXNET
    else:
        print("Mobile")
        return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device), NetType.MobileNet

def get_AlexSqueeze(rand_client):
    p = rand_client.random()
    if p <= 0.5:
        print("Alex")
        return AlexNet(num_classes=experiment_config.num_classes).to(device),NetType.ALEXNET
    else:
        print("SqueezeNetServer")
        return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device), NetType.SqueezeNet

def get_AlexMobileResnet(rand_client):
    p = rand_client.random()
    if p <= 0.3:
        print("Alex")
        return AlexNet(num_classes=experiment_config.num_classes).to(device),NetType.ALEXNET
    elif 0.3<p <= 0.6:
        print("Mobile")
        return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device), NetType.MobileNet
    else:
        print("Res")
        return ResNet18Server(num_classes=experiment_config.num_classes).to(device),NetType.ResNet

def get_server_model():
    if experiment_config.net_cluster_technique== NetClusterTechnique.multi_head:
        num_heads = experiment_config.num_clusters
        if isinstance(num_heads, str):
            num_heads = experiment_config.number_of_optimal_clusters
    else:
        num_heads = 1

    if experiment_config.server_net_type == NetType.ALEXNET:
        return AlexNet(num_classes=experiment_config.num_classes,num_clusters=num_heads).to(device)
    if experiment_config.server_net_type == NetType.VGG:
        return VGGServer(num_classes=experiment_config.num_classes,num_clusters=num_heads).to(device)
    if experiment_config.server_net_type == NetType.MobileNet:
        return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device)
    if experiment_config.server_net_type == NetType.DenseNetServer:
        return DenseNetServer(num_classes=experiment_config.num_classes,num_clusters=num_heads).to(device)


class LearningEntity(ABC):
    def __init__(self,id_,global_data,test_global_data):
        self.test_global_data= test_global_data
        self.seed =experiment_config.seed_num
        self.global_data = global_data
        self.pseudo_label_received = {}
        self.pseudo_label_to_send = None
        self.current_iteration = 0
        self.epoch_count = 0
        self.id_ = id_
        self.model=None
        #self.weights = None
        self.accuracy_per_client_1 = {}
        self.accuracy_per_client_10= {}
        self.accuracy_per_client_100= {}
        self.accuracy_per_client_5= {}
        self.size_sent = {}
        #self.accuracy_per_client_5 = {}

        #self.accuracy_pl_measures= {}
        #self.accuracy_test_measures_k_half_cluster = {}
        #self.accuracy_pl_measures_k_half_cluster= {}


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

    #def set_weights(self):
        #if self.weights  is None:
        #    self.model.apply(self.initialize_weights)
        #else:
        #    self.model.apply(self.weights)

    import torch
    import torch.nn.functional as F

    def apply_temperature_to_probs(self, pseudo_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature T to probability vectors.

        - T = 1.0  -> no change
        - T < 1.0  -> sharpen (harder labels)
        - T > 1.0  -> smooth (softer labels)
        - T = 0.0  -> HARD labels (argmax -> one-hot)
        """
        T = getattr(experiment_config, "distill_temperature", 1.0)

        # Basic sanity / normalization
        pseudo_probs = pseudo_probs.clamp_min(1e-8)
        pseudo_probs = pseudo_probs / pseudo_probs.sum(dim=1, keepdim=True)

        # Handle T = 0 → hard labels (one-hot)
        if T is not None and abs(T) < 1e-8:
            # pseudo_probs: [batch_size, num_classes]
            hard_idx = torch.argmax(pseudo_probs, dim=1)  # [batch_size]
            num_classes = pseudo_probs.size(1)
            hard_one_hot = F.one_hot(hard_idx, num_classes=num_classes).float()
            return hard_one_hot

        # T = 1 → no change
        if T is None or abs(T - 1.0) < 1e-6:
            return pseudo_probs

        # General case: p_T ∝ p^(1/T)
        pseudo_probs_T = pseudo_probs.pow(1.0 / T)
        pseudo_probs_T = pseudo_probs_T / pseudo_probs_T.sum(dim=1, keepdim=True)

        return pseudo_probs_T

    def get_pseudo_label_L2(self,pseudo_labels):
        loader = DataLoader(self.global_data, batch_size=len(self.global_data))
        X_tensor, Y_tensor = next(iter(loader))  # Gets all data
        # Convert to NumPy (if needed)
        ground_truth = Y_tensor.numpy()

        num_classes = pseudo_labels.shape[1]
        ground_truth_onehot = F.one_hot(torch.tensor(ground_truth), num_classes=num_classes).float().numpy()

        return  np.mean(np.linalg.norm(pseudo_labels - ground_truth_onehot, axis=1) ** 2)
    def iterate(self,t):
        #self.set_weights()
        torch.manual_seed(self.num+t*17)
        torch.cuda.manual_seed(self.num+t*17)

        os.makedirs("./saved_models", exist_ok=True)

        if experiment_config.is_with_memory_load and self.id_ != "server":
            if t == 0:
                #self.model = self.get_client_model()
                self.model= self.get_model_by_type()
                self.model.apply(self.initialize_weights)
                self.iteration_context(t)
                if experiment_config.is_with_memory_load:
                    torch.save(self.model.state_dict(), "./saved_models/model_{}.pth".format(self.id_))
                    del self.model
            elif t > 0:
                #self.model = self.get_client_model()
                self.model = self.get_model_by_type()
                self.model.apply(self.initialize_weights)
                if experiment_config.is_with_memory_load:
                    self.model.load_state_dict(torch.load("./saved_models/model_{}.pth".format(self.id_)))
                self.iteration_context(t)
                if experiment_config.is_with_memory_load:
                    torch.save(self.model.state_dict(), "./saved_models/model_{}.pth".format(self.id_))
                del self.model
        else:
            self.iteration_context(t)



        #if isinstance(self,Client):
        #    self.loss_measures[t]=self.evaluate_test_loss()
        #    self.accuracy_test_measures[t]=self.evaluate_accuracy(self.test_set)
        #    self.accuracy_pl_measures[t]=self.evaluate_accuracy(self.global_data)
        #    self.accuracy_test_measures_k_half_cluster[t]=self.evaluate_accuracy(self.test_set,self.model)
        #    self.accuracy_pl_measures_k_half_cluster[t]=self.evaluate_accuracy(self.global_data,self.model)



        #if isinstance(self,Server) and experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_cluster:
            #raise Exception("need to evaluate use backbone")
        #    for cluster_id in range(experiment_config.num_clusters):
        #        self.loss_measures[cluster_id][t] = self.evaluate_test_loss(cluster_id=cluster_id, model=None)
        #        self.accuracy_test_measures[cluster_id][t] = self.evaluate_accuracy(self.test_set,model= None,k=1,cluster_id=cluster_id )
        #        self.accuracy_pl_measures[cluster_id][t] = self.evaluate_accuracy(self.global_data,model= None,k=1,cluster_id=cluster_id )
        #        self.accuracy_test_measures_k_half_cluster[cluster_id][t] = self.evaluate_accuracy(self.test_set ,model= None, k = experiment_config.num_clusters // 2,cluster_id=cluster_id )
        #        self.accuracy_pl_measures_k_half_cluster[cluster_id][t] = self.evaluate_accuracy(self.global_data,model= None, k =experiment_config.num_clusters // 2,cluster_id=cluster_id )
    @abstractmethod
    def iteration_context(self,t):
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

       #print(f"Shape of the 2D pseudo-label matrix: {all_probs.shape}")
        return all_probs

    def evaluate_max_accuracy_per_point(self,models, data_, k=1, cluster_id=None):
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

    def get_model_by_type(self):

        if self.model_type  == NetType.ALEXNET:
            return AlexNet(num_classes=experiment_config.num_classes).to(device)
        if self.model_type  == NetType.VGG:
            return VGGServer(num_classes=experiment_config.num_classes).to(device)
        if self.model_type  == NetType.ResNet:
            return ResNet18Server(num_classes=experiment_config.num_classes).to(device)
        if self.model_type  == NetType.SqueezeNet:
            return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device)

        if self.model_type == NetType.MobileNet:
            return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device)


class Client(LearningEntity):
    def __init__(self, id_, client_data, global_data,global_test_data,local_test_data):
        LearningEntity.__init__(self,id_,global_data,global_test_data)



        self.num = (self.id_+1)*17
        self.local_test_set= local_test_data

        self.local_data = client_data
        self.epoch_count = 0
        self.seed_client =experiment_config.seed_num + (self.num+1)*17
        self.rand_client =Random(self.seed_client)

        self.model, self.model_type = self.get_client_model()
        self.model.apply(self.initialize_weights)
        #self.train_learning_rate = experiment_config.learning_rate_train_c
        #self.weights = None
        self.global_data =global_data
        self.server = None
        self.pseudo_label_L2 = {}
        self.global_label_distribution = self.get_label_distribution()

    def get_client_model(self):
        if experiment_config.client_net_type == NetType.ALEXNET:
            return AlexNet(num_classes=experiment_config.num_classes).to(device), NetType.ALEXNET
        if experiment_config.client_net_type == NetType.VGG:
            return VGGServer(num_classes=experiment_config.num_classes).to(device), NetType.VGG
        if experiment_config.client_net_type == NetType.ResNet:
            return ResNet18Server(num_classes=experiment_config.num_classes).to(device), NetType.ResNet
        if experiment_config.client_net_type == NetType.SqueezeNet:
            return SqueezeNetServer(num_classes=experiment_config.num_classes).to(device), NetType.SqueezeNet

        if experiment_config.client_net_type == NetType.MobileNet:
            return MobileNetV2Server(num_classes=experiment_config.num_classes).to(device), NetType.MobileNet

        if experiment_config.client_net_type == NetType.ResNetSqueeze:
            return get_ResNetSqueeze(self.rand_client)
        if experiment_config.client_net_type == NetType.ResMobile:
            return get_ResNetMobile(self.rand_client)
        if experiment_config.client_net_type == NetType.AlexMobile:
            return get_AlexMobile(self.rand_client)
        if experiment_config.client_net_type == NetType.rndStrong:
            return get_rnd_strong_net(self.rand_client)
        if experiment_config.client_net_type == NetType.rndWeak:
            return get_rnd_weak_net(self.rand_client)
        if experiment_config.client_net_type == NetType.rndNet:
            return get_rnd_net(self.rand_client)
        if experiment_config.client_net_type == NetType.AlexSqueeze:
            return get_AlexSqueeze(self.rand_client)
        if experiment_config.client_net_type == NetType.AlexMobileResnet:
            return get_AlexMobileResnet(self.rand_client)

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
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)
                pseudo_targets = self.apply_temperature_to_probs(pseudo_targets)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(f"Skipping batch {batch_idx}: Pseudo target size mismatch.")
                    continue
                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN/Inf in pseudo targets at batch {batch_idx}")
                    continue

                #pseudo_targets = F.softmax(pseudo_targets, dim=1)

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
            if t>0:
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

            train_loss = self.fine_tune()

            self.pseudo_label_to_send = self.evaluate()
            what_to_send = self.pseudo_label_to_send


            self.size_sent[t] = (what_to_send.numel() * what_to_send.element_size()) / (1024 * 1024)
            self.pseudo_label_L2[t] = self.get_pseudo_label_L2(what_to_send)
            acc = self.evaluate_accuracy_single(self.local_test_set)

            acc_test = self.evaluate_accuracy_single(self.test_global_data)
            if experiment_config.data_set_selected == DataSet.CIFAR100:
                if acc != 1 and acc_test!=1:
                    break
                else:
                    self.model.apply(self.initialize_weights)
            if experiment_config.data_set_selected == DataSet.CIFAR10 or experiment_config.data_set_selected == DataSet.SVHN :
                if acc != 10 and acc_test!=10:
                    break
                else:
                    self.model.apply(self.initialize_weights)
            if experiment_config.data_set_selected == DataSet.TinyImageNet or experiment_config.data_set_selected == DataSet.ImageNetR:
                if acc != 0.5 and acc_test !=0.5:
                    break
                else:
                    self.model.apply(self.initialize_weights)

            if experiment_config.data_set_selected == DataSet.EMNIST_balanced:
                if acc > 2.14 and acc_test>2.14:
                    break
                else:
                    self.model.apply(self.initialize_weights)
        print("hi")


        #self.print_grad_size()


        self.accuracy_per_client_1[t] = self.evaluate_accuracy_single(self.local_test_set, k=1)
        self.accuracy_per_client_10[t] = self.evaluate_accuracy(self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)

        self.accuracy_per_client_5[t] = self.evaluate_accuracy(self.local_test_set, k=5)


    def train_with_consistency(self, mean_pseudo_labels):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")
        print(f"*** {self.__str__()} train ***")

        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False,
                                   num_workers=0, drop_last=True)
        self.model.train()

        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        lambda_consistency = experiment_config.lambda_consistency # You can tune this
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
                pseudo_targets = self.apply_temperature_to_probs(pseudo_targets)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(f"Skipping batch {batch_idx}: Pseudo target size mismatch.")
                    continue
                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN/Inf in pseudo targets at batch {batch_idx}")
                    continue

                #pseudo_targets = F.softmax(pseudo_targets, dim=1)
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

    def get_global_label_distribution(self, k ):
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
                pseudo_targets = self.apply_temperature_to_probs(pseudo_targets)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}")
                    continue

                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN or Inf found in pseudo targets at batch {batch_idx}: {pseudo_targets}")
                    continue

                #pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # Get weights based on true labels
                weights = torch.tensor(
                    [    self.get_global_label_distribution(label.item()) /len(self.global_data) for label in true_labels],
                    dtype=torch.float32, device=device
                ).unsqueeze(1)  # (batch_size, 1)

                # KL divergence per sample
                loss_per_sample = F.kl_div(outputs_prob, pseudo_targets, reduction='none').sum(dim=1)
                weighted_loss = (loss_per_sample * weights.squeeze()).mean()
                loss = weighted_loss

                #print("Type of loss:", type(loss))
                #print("Loss value:", loss)

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


    def reinit_last_linear(self):
        last_linear = None
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            print("[warn] No nn.Linear layer found to reinit.")
            return

        nn.init.normal_(last_linear.weight, mean=0.0, std=0.02)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)
        print(f"[info] Reinitialized layer: {last_linear}")

    def train(self,mean_pseudo_labels):
        #self.reinit_last_linear()
        print(f"cccccccccccc_Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False, num_workers=0,
                                   drop_last=True)
        #server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False,
        #                           num_workers=0)
        #print(1)
        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam( self.model.parameters(), lr=experiment_config.learning_rate_train_c)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c,
        #                             weight_decay=1e-4)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train_client):
            #print(2)

            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                #print(batch_idx)

                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs =  self.model(inputs)
                # Check for NaN or Inf in outputs

                # Convert model outputs to log probabilities
                outputs_prob = F.log_softmax(outputs, dim=1)
                # Slice pseudo_targets to match the input batch size
                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)
                pseudo_targets = self.apply_temperature_to_probs(pseudo_targets)

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
                #pseudo_targets = F.softmax(pseudo_targets, dim=1)

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

        #self.weights =self.model.state_dict()
        return avg_loss


    def train__(self, mean_pseudo_labels):
        print(f"bbbbbbbbbbbbbbbb*** {self.__str__()} train ***")
        print(f"Mean pseudo-labels shape: {tuple(mean_pseudo_labels.shape)}")

        server_loader = DataLoader(
            self.global_data,
            batch_size=experiment_config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=experiment_config.learning_rate_train_c)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        # ---- One-time diagnostics ----
        with torch.no_grad():
            pmin = pseudo_targets_all.min().item()
            pmax = pseudo_targets_all.max().item()
            row_sums = pseudo_targets_all.sum(dim=1)
            print(f"[diag] pseudo_labels min/max: {pmin:.6f}/{pmax:.6f}")
            print(f"[diag] row_sums (first 5): {[float(x) for x in row_sums[:5]]}")
            pt = torch.clamp(pseudo_targets_all, 1e-8, None)
            pt = pt / pt.sum(dim=1, keepdim=True)
            ent = -(pt * torch.log(pt)).sum(dim=1).mean().item()
            print(f"[diag] mean target entropy (safe-renormed): {ent:.6f}")

            # Optional: measure pre-update agreement to confirm trivial KL
            try:
                probe_inputs, _ = next(iter(server_loader))
                probe_inputs = probe_inputs.to(device)
                logits0 = self.model(probe_inputs)
                logprob0 = F.log_softmax(logits0, dim=1)
                prob0 = logprob0.exp()
                bs = probe_inputs.size(0)
                pt0 = pseudo_targets_all[:bs]
                sums = pt0.sum(dim=1)
                if torch.allclose(sums.mean(), torch.tensor(1.0, device=device), atol=1e-3):
                    pt0 = torch.clamp(pt0, 1e-8);
                    pt0 = pt0 / pt0.sum(dim=1, keepdim=True)
                else:
                    pt0 = F.softmax(pt0, dim=1)
                kl0 = F.kl_div(logprob0, pt0, reduction="batchmean").item()
                agree0 = (prob0.argmax(dim=1) == pt0.argmax(dim=1)).float().mean().item()
                print(f"[pre] KL(student||pseudo): {kl0:.6f}  agree: {agree0:.3f}")
            except StopIteration:
                pass

        num_epochs = experiment_config.epochs_num_train_client
        last_avg_loss = float('nan')

        for epoch in range(num_epochs):
            self.epoch_count += 1
            epoch_loss = 0.0
            processed = 0

            cursor = 0
            total_rows = pseudo_targets_all.size(0)

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                bs = inputs.size(0)

                start_idx = cursor
                end_idx = min(cursor + bs, total_rows)
                cursor = end_idx

                if end_idx - start_idx != bs:
                    print(f"[warn] batch {batch_idx}: pseudo targets short "
                          f"({end_idx - start_idx} vs {bs}). Skipping.")
                    continue

                pseudo_targets = pseudo_targets_all[start_idx:end_idx]

                # Decide if targets are probabilities or logits
                with torch.no_grad():
                    sums = pseudo_targets.sum(dim=1)
                    already_probs = torch.allclose(
                        sums.mean(),
                        torch.tensor(1.0, device=sums.device),
                        atol=1e-3
                    )

                if already_probs:
                    pseudo_targets = torch.clamp(pseudo_targets, min=1e-8)
                    pseudo_targets = pseudo_targets / pseudo_targets.sum(dim=1, keepdim=True)
                else:
                    pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # ---- Make targets informative: sharpen + light smoothing ----
                T = 0.5
                pseudo_targets = pseudo_targets.clamp_min(1e-8).pow(1.0 / T)
                pseudo_targets = pseudo_targets / pseudo_targets.sum(dim=1, keepdim=True)
                alpha = 0.05
                num_classes = pseudo_targets.size(1)
                uniform = torch.full_like(pseudo_targets, 1.0 / num_classes)
                pseudo_targets = (1 - alpha) * pseudo_targets + alpha * uniform

                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"[warn] NaN/Inf in pseudo targets (batch {batch_idx}). Skipping.")
                    continue

                optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs_logprob = F.log_softmax(outputs, dim=1)

                loss = criterion(outputs_logprob, pseudo_targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[warn] NaN/Inf loss (batch {batch_idx}). Skipping.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                processed += 1

            if processed == 0:
                print("[error] No batches processed this epoch — check alignment between "
                      "`global_data` and `mean_pseudo_labels` (sizes/order).")
                last_avg_loss = float('nan')
            else:
                last_avg_loss = epoch_loss / processed

            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Processed {processed}/{len(server_loader)} batches, "
                  f"Loss: {last_avg_loss:.6f}")

        return last_avg_loss

    def __str__(self):
        return "Client " + str(self.id_)

    def fine_tune(self):
        print("*** " + self.__str__() + " fine-tune ***")

        # Load the weights into the model
        #if self.weights is  None:
        #    self.model.apply(self.initialize_weights)
        #else:
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
        #self.weights = self.model.state_dict()self.weights = self.model.state_dict()

        return  result_to_print

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



class Server(LearningEntity):
    def __init__(self,id_,global_data,test_data, clients_ids,clients_test_data_dict):
        LearningEntity.__init__(self, id_,global_data,test_data)
        #self.local_batch = experiment_config.local_batch
        self.pseudo_label_before_net_L2 = {}
        self.pseudo_label_after_net_L2 = {}
        self.num = (1000)*17
        self.pseudo_label_received = {}
        self.clusters_client_id_dict_per_iter = {}
        self.clients_ids = clients_ids
        self.reset_clients_received_pl()
        self.clients_test_data_dict=clients_test_data_dict
        #if experiment_config.server_learning_technique == ServerLearningTechnique.multi_head:
        self.accuracy_per_client_1_max = {}

        self.previous_centroids_dict = {}
        self.pseudo_label_to_send = {}
        if isinstance(experiment_config.num_clusters,int):
            num_clusters = experiment_config.num_clusters
        else:
            num_clusters = experiment_config.number_of_optimal_clusters

        self.accuracy_server_test_1 = {}
        self.accuracy_global_data_1 = {}
        self.accuracy_test_max = {}
        self.accuracy_global_max = {}

        if num_clusters>0:
            for cluster_id in  range(num_clusters):
                self.previous_centroids_dict[cluster_id] = None

        if  experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
            self.model = get_server_model()
            self.model.apply(self.initialize_weights)


        if  experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            self.multi_model_dict = {}

            if num_clusters>0:
                for cluster_id in  range(num_clusters):
                    self.multi_model_dict[cluster_id] = get_server_model()
                    self.multi_model_dict[cluster_id].apply(self.initialize_weights)


        self.accuracy_per_client_1_max = {}
        self.accuracy_per_client_10_max= {}
        self.accuracy_per_client_100_max= {}
        self.accuracy_per_client_5_max= {}

        for client_id in self.clients_ids:
            self.accuracy_per_client_1[client_id] = {}
            self.accuracy_per_client_1_max[client_id] = {}
            self.accuracy_per_client_5_max[client_id] = {}
            self.accuracy_per_client_10_max[client_id] = {}
            self.accuracy_per_client_100_max[client_id] = {}

        if num_clusters>0:
            for cluster_id in  range(num_clusters):
                self.accuracy_server_test_1[cluster_id] = {}
                self.accuracy_global_data_1[cluster_id] ={}
        #self.accuracy_aggregated_head = {}
        #self.accuracy_pl_measures[cluster_id] = {}



        #self.model = get_server_model()
        #self.model.apply(self.initialize_weights)
        #self.train_learning_rate = experiment_config.learning_rate_train_s
        #self.train_epoches = experiment_config.epochs_num_train_server

        #self.weights = None





    def receive_single_pseudo_label(self, sender, info):
        self.pseudo_label_received[sender] = info

    def get_pseudo_label_list_after_models_train(self,mean_pseudo_labels_per_cluster):
        pseudo_labels_for_model_per_cluster = {}
        for cluster_id, mean_pseudo_label_for_cluster in mean_pseudo_labels_per_cluster.items():
            model_ = self.model_per_cluster[cluster_id]
            self.train(mean_pseudo_label_for_cluster, model_, str(cluster_id))
            pseudo_labels_for_model = self.evaluate(model_)
            pseudo_labels_for_model_per_cluster[cluster_id] = pseudo_labels_for_model
        ans = list(pseudo_labels_for_model_per_cluster.values())
        return ans

    def select_confident_pseudo_labels(self, cluster_pseudo_labels):
        """
        Select pseudo-labels from the cluster with the highest confidence for each data point.

        Args:
            cluster_pseudo_labels (list of torch.Tensor): List of tensors where each tensor contains pseudo-labels
                                                          from a cluster with shape [num_data_points, num_classes].

        Returns:
            torch.Tensor: A tensor containing the selected pseudo-labels of shape [num_data_points, num_classes].
        """
        num_clusters = len(cluster_pseudo_labels)
        num_data_points = cluster_pseudo_labels[0].size(0)

        # Store the maximum confidence and the corresponding cluster index for each data point
        max_confidences = torch.zeros(num_data_points, device=cluster_pseudo_labels[0].device)
        selected_labels = torch.zeros_like(cluster_pseudo_labels[0])

        for cluster_idx, pseudo_labels in enumerate(cluster_pseudo_labels):
            # Compute the max confidence for the current cluster
            cluster_max_confidences, _ = torch.max(pseudo_labels, dim=1)

            # Update selected labels where the current cluster has higher confidence
            mask = cluster_max_confidences > max_confidences
            max_confidences[mask] = cluster_max_confidences[mask]
            selected_labels[mask] = pseudo_labels[mask]

        return selected_labels
    def create_feed_back_to_clients_multihead(self,mean_pseudo_labels_per_cluster,t):
        for _ in range(experiment_config.num_rounds_multi_head):
            for cluster_id, mean_pseudo_label_for_cluster in mean_pseudo_labels_per_cluster.items():
                self.train(mean_pseudo_label_for_cluster, cluster_id)
            if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_cluster:

                for cluster_id in mean_pseudo_labels_per_cluster.keys():
                        pseudo_labels_for_cluster = self.evaluate_for_cluster(cluster_id)
                        for client_id in self.clusters_client_id_dict_per_iter[t][cluster_id]:
                            self.pseudo_label_to_send[client_id] = pseudo_labels_for_cluster

        if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_client:
            pseudo_labels_for_cluster_list = []
            for cluster_id, mean_pseudo_label_for_cluster in mean_pseudo_labels_per_cluster.items():
                self.train(mean_pseudo_label_for_cluster, cluster_id)
            for cluster_id in mean_pseudo_labels_per_cluster.keys():
                pseudo_labels_for_cluster_list.append(self.evaluate_for_cluster(cluster_id))

            pseudo_labels_to_send = self.select_confident_pseudo_labels(pseudo_labels_for_cluster_list)

            for client_id in self.clients_ids:
                self.pseudo_label_to_send[client_id] = pseudo_labels_to_send

    def create_feed_back_to_clients_multimodel(self,mean_pseudo_labels_per_cluster,t):
        pl_per_cluster = {}

        for cluster_id, mean_pseudo_label_for_cluster in mean_pseudo_labels_per_cluster.items():
            print("cluster_id",cluster_id)
            selected_model = self.multi_model_dict[cluster_id]
            for _ in range(5):

                if experiment_config.input_consistency == InputConsistency.withInputConsistency:
                    self.train_with_consistency(mean_pseudo_label_for_cluster, 0,selected_model)
                else:
                    self.train(mean_pseudo_label_for_cluster, 0,selected_model)

                acc_global = self.evaluate_accuracy_single(self.test_global_data, model=selected_model, k=1,
                                              cluster_id=0)

                acc_local = self.evaluate_accuracy_single(self.global_data, model=selected_model, k=1,
                                                           cluster_id=0)

                if experiment_config.data_set_selected == DataSet.CIFAR100:
                    if acc_global != 1 and acc_local != 1:
                        break
                    else:
                        selected_model.apply(self.initialize_weights)
                if experiment_config.data_set_selected == DataSet.CIFAR10 or experiment_config.data_set_selected == DataSet.SVHN:
                    if acc_global != 10 and acc_local != 10:
                        break
                    else:
                        selected_model.apply(self.initialize_weights)
                if experiment_config.data_set_selected == DataSet.TinyImageNet:
                    if acc_global != 0.5 and acc_local != 0.5:
                        break
                    else:
                        selected_model.apply(self.initialize_weights)

                if experiment_config.data_set_selected == DataSet.EMNIST_balanced:
                    if acc_global > 2.14 and acc_local > 2.14:
                        break
                    else:
                        selected_model.apply(self.initialize_weights)

            print("hihi")
            if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_cluster:
                pseudo_labels_for_cluster = self.evaluate_for_cluster(0,selected_model)
                pl_per_cluster[cluster_id] = pseudo_labels_for_cluster
                for client_id in self.clusters_client_id_dict_per_iter[t][cluster_id]:
                    self.pseudo_label_to_send[client_id] = pseudo_labels_for_cluster
        if experiment_config.server_feedback_technique == ServerFeedbackTechnique.similar_to_client:
            pseudo_labels_for_cluster_list = []
            for cluster_id, mean_pseudo_label_for_cluster in mean_pseudo_labels_per_cluster.items():
                pl_per_cluster[cluster_id] = mean_pseudo_label_for_cluster

                selected_model = self.multi_model_dict[cluster_id]
                self.train(mean_pseudo_label_for_cluster, 0, selected_model)
                pl = self.evaluate_for_cluster(0,selected_model)

                pseudo_labels_for_cluster_list.append(pl)

            pseudo_labels_to_send = self.select_confident_pseudo_labels(pseudo_labels_for_cluster_list)

            for client_id in self.clients_ids:
                self.pseudo_label_to_send[client_id] = pseudo_labels_to_send

        return pl_per_cluster
    def evaluate_results(self,t):

        if isinstance(experiment_config.num_clusters,int):
            num_clusters =experiment_config.num_clusters
        else:
            num_clusters =experiment_config.number_of_optimal_clusters
        models_list = list(self.multi_model_dict.values())
        self.accuracy_global_data_1[t] = self.evaluate_max_accuracy_per_point(models = models_list, data_ = self.global_data, k=1, cluster_id=None)
        for cluster_id in range(num_clusters):
            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                selected_model = self.multi_model_dict[cluster_id]
            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
                selected_model = None
            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                cluster_id_to_examine = 0
            else: cluster_id_to_examine = cluster_id
            self.accuracy_server_test_1[cluster_id][t] = self.evaluate_accuracy_single(self.test_global_data,
                                                                                model=selected_model, k=1,
                                                                                cluster_id=cluster_id_to_examine)


            #self.accuracy_global_data_1[cluster_id][t] = self.evaluate_accuracy(self.global_data,
            #                                                                    model=selected_model, k=1,
            #                                                                    cluster_id=cluster_id_to_examine)

        for client_id in self.clients_ids:
            test_data_per_clients = self.clients_test_data_dict[client_id]
            cluster_id_for_client = self.get_cluster_of_client(client_id, t)
            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                print("cluster_id_for_client",cluster_id_for_client)
                print("len(self.multi_model_dict)",len(self.multi_model_dict))

                selected_model = self.multi_model_dict[cluster_id_for_client]
                cluster_id_for_client = 0

            if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
                selected_model = None
            print("client_id",client_id,"accuracy_per_client_1")
            self.accuracy_per_client_1[client_id][t] = self.evaluate_accuracy_single(test_data_per_clients,
                                                                              model=selected_model, k=1,
                                                                              cluster_id=cluster_id_for_client)
            #print("client_id",client_id,"accuracy_per_client_5")
            #self.accuracy_per_client_5[client_id][t] = self.evaluate_accuracy(test_data_per_clients,
             #                                                                 model=selected_model, k=5,
             #                                                                 cluster_id=cluster_id_for_client)
            l1 = []
            l2 = []
            l3 = []
            l4 = []


            for cluster_id in range(num_clusters):
                if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
                    l1.append(self.evaluate_accuracy_single(self.clients_test_data_dict[client_id], model=self.multi_model_dict[cluster_id], k=1,
                                                     cluster_id=0))
                    l2.append(self.evaluate_accuracy(self.clients_test_data_dict[client_id],
                                                     model=self.multi_model_dict[cluster_id], k=10,
                                                     cluster_id=0))
                    l3.append(self.evaluate_accuracy(self.clients_test_data_dict[client_id],
                                                     model=self.multi_model_dict[cluster_id], k=100,
                                                     cluster_id=0))
                    l4.append(self.evaluate_accuracy(self.clients_test_data_dict[client_id],
                                                     model=self.multi_model_dict[cluster_id], k=5,
                                                     cluster_id=0))

                else:
                    l1.append(self.evaluate_accuracy(self.clients_test_data_dict[client_id], model=selected_model, k=1,
                                                    cluster_id=cluster_id))

            print("client_id",client_id,"accuracy_per_client_1_max",max(l1))

            self.accuracy_per_client_1_max[client_id][t] = max(l1)
            self.accuracy_per_client_10_max[client_id][t] = max(l2)
            self.accuracy_per_client_100_max[client_id][t] = max(l3)

            self.accuracy_per_client_5_max[client_id][t] = max(l4)

    def iteration_context(self,t):
        self.current_iteration = t
        pseudo_labels_per_cluster, self.clusters_client_id_dict_per_iter[t] = self.get_pseudo_labels_input_per_cluster(t)  # #
        self.pseudo_label_before_net_L2[t]={}
        for cluster_id, pl in pseudo_labels_per_cluster.items():
            self.pseudo_label_before_net_L2[t][cluster_id] = self.get_pseudo_label_L2(pl)

        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_head:
            self.create_feed_back_to_clients_multihead(pseudo_labels_per_cluster,t)
        if experiment_config.net_cluster_technique == NetClusterTechnique.multi_model:
            pseudo_labels_for_cluster = self.create_feed_back_to_clients_multimodel(pseudo_labels_per_cluster,t)

        self.pseudo_label_after_net_L2[t] = {}
        #for cluster_id, pseudo_labels_per_client in pseudo_labels_for_cluster.items():





        loader = DataLoader(self.global_data, batch_size=len(self.global_data))
        X_tensor, Y_tensor = next(iter(loader))  # Gets all data
        # Convert to NumPy (if needed)
        ground_truth = Y_tensor.numpy()
        num_classes = 100

        #gt_onehot = F.one_hot(torch.tensor(ground_truth), num_classes=num_classes).float().numpy()  # (N, C)

        # Stack into a tensor of shape (K, N, C)
        #pseudo_stack = np.stack(pseudo_labels_for_cluster.values())  # (K, N, C)

        # Compute squared L2 distances to one-hot ground truth → shape: (K, N)
        #l2_errors = np.linalg.norm(pseudo_stack - gt_onehot[None, :, :], axis=2) ** 2

        # Take min L2 error for each data point → shape: (N,)
        #min_errors = np.min(l2_errors, axis=0)

        self.pseudo_label_after_net_L2[t] = 0

        self.evaluate_results(t)
        self.reset_clients_received_pl()

    def train_with_consistency(self, mean_pseudo_labels, cluster_num="0", selected_model=None):
        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)
        print(f"*** {self.__str__()} train *** Cluster: {cluster_num} ***")

        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False,
                                   num_workers=0, drop_last=True)

        if selected_model is None:
            selected_model_train = self.model
        else:
            selected_model_train = selected_model

        selected_model_train.train()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_consistency = nn.MSELoss()
        lambda_consistency = experiment_config.lambda_consistency  # You can tune this hyperparameter

        optimizer = torch.optim.Adam(selected_model_train.parameters(), lr=experiment_config.learning_rate_train_s)
        pseudo_targets_all = mean_pseudo_labels.to(device)

        def add_noise(inputs, std=0.05):
            noise = torch.randn_like(inputs) * std
            return torch.clamp(inputs + noise, 0., 1.)

        for epoch in range(experiment_config.epochs_num_train_server):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                # Original model output
                outputs = selected_model_train(inputs, cluster_id=cluster_num)
                outputs_prob = F.log_softmax(outputs, dim=1)

                # Slice pseudo-targets
                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)
                pseudo_targets = self.apply_temperature_to_probs(pseudo_targets)

                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}")
                    continue

                #pseudo_targets = F.softmax(pseudo_targets, dim=1)

                loss_kl = criterion_kl(outputs_prob, pseudo_targets)

                # Input consistency regularization
                inputs_aug = add_noise(inputs)
                with torch.no_grad():
                    outputs_aug = selected_model_train(inputs_aug, cluster_id=cluster_num)
                    probs = F.softmax(outputs, dim=1)
                    probs_aug = F.softmax(outputs_aug, dim=1)

                loss_consistency = criterion_consistency(probs, probs_aug)

                # Combine losses
                loss = loss_kl + lambda_consistency * loss_consistency

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(selected_model_train.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train_server}], Loss: {avg_loss:.4f}")

        return avg_loss
    def train(self, mean_pseudo_labels,  cluster_num="0",selected_model=None):

        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train *** Cluster: {cluster_num} ***")

        #experiment_config.batch_size
        #self.local_batch = 32
        server_loader = DataLoader(self.global_data , batch_size=experiment_config.batch_size, shuffle=False,
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
                pseudo_targets = self.apply_temperature_to_probs(pseudo_targets)

                # Check if pseudo_targets size matches the input batch size
                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}"
                    )
                    continue

                # Normalize pseudo targets to sum to 1
                #pseudo_targets = F.softmax(pseudo_targets, dim=1)

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


    def reset_clients_received_pl(self):
        for id_ in self.clients_ids:
            self.pseudo_label_received[id_] = None



    def k_means_grouping(self):
        """
        Groups agents into k clusters based on the similarity of their pseudo-labels,
        with memory of cluster centroids from the previous iteration.

        Args:
            k (int): Number of clusters for k-means.

        Returns:
            dict: A dictionary where keys are cluster indices (0 to k-1) and values are lists of client IDs in that cluster.
        """
        k = experiment_config.num_clusters
        client_data = self.pseudo_label_received

        # Extract client IDs and pseudo-labels
        client_ids = list(client_data.keys())
        pseudo_labels = [client_data[client_id].flatten().numpy() for client_id in client_ids]

        # Stack pseudo-labels into a matrix for clustering
        data_matrix = np.vstack(pseudo_labels)

        # Prepare initial centroids based on previous clusters if available

        if not self.centroids_are_empty():
            # Convert previous centroids from a dictionary to a 2D array for KMeans init
            # Assuming `self.previous_centroids` is a dictionary with cluster indices as keys and centroids as values.
            # We need to extract the centroids as a list of numpy arrays and ensure it has shape (k, n_features).
            previous_centroids_array = np.array(list(self.previous_centroids_dict.values()))

            # Check if the number of centroids matches k
            if previous_centroids_array.shape[0] != k:
                raise ValueError(f"Previous centroids count does not match the number of clusters (k={k}).")

            # Initialize k-means with the previous centroids
            kmeans = KMeans(n_clusters=k, init=previous_centroids_array, n_init=1, random_state=42)
        else:
            # Default initialization
            kmeans = KMeans(n_clusters=k, random_state=42)

        # Perform clustering
        kmeans.fit(data_matrix)

        # Update centroids for the next iteration
        self.previous_centroids_dict = {i: centroid for i, centroid in enumerate(kmeans.cluster_centers_)}

        # Assign clients to clusters
        cluster_assignments = kmeans.predict(data_matrix)
        clusters = {i: [] for i in range(k)}

        # Assign client IDs to their respective clusters
        for client_id, cluster in zip(client_ids, cluster_assignments):
            clusters[cluster].append(client_id)

        return clusters

    def get_cluster_mean_pseudo_labels_dict(self,clusters_client_id_dict):
        ans = {}
        for cluster_id, clients_ids in clusters_client_id_dict.items():
            ans[cluster_id] = []
            for client_id in clients_ids:
                ans[cluster_id].append(self.pseudo_label_received[client_id])
        return ans

    def calc_L2(self,pair):
        first_pl = self.pseudo_label_received[pair[0]]
        second_pl = self.pseudo_label_received[pair[1]]


        return  Server.calc_L2_given_pls(first_pl,second_pl)
    #def get_L2_of_all_clients(self):

        # Example list of client IDs
        #pairs = list(combinations(self.clients_ids, 2))
        #ans_dict = {}
        #for pair in pairs:
        #    ans_dict[pair] = self.calc_L2(pair).item()
        #return ans_dict

    @staticmethod
    def calc_L2_given_pls(pl1,pl2):
        difference = pl1 - pl2  # Element-wise difference
        squared_difference = difference ** 2  # Square the differences
        sum_squared = torch.sum(squared_difference)  # Sum of squared differences
        return torch.sqrt(sum_squared).item()  # Take the square root
    def initiate_clusters_centers_dict(self,L2_of_all_clients):
        max_pair = max(L2_of_all_clients.items(), key=lambda item: item[1])
        max_pair_keys = max_pair[0]
        clusters_centers_dict = {max_pair_keys[0]: self.pseudo_label_received[max_pair_keys[0]],
                                 max_pair_keys[1]: self.pseudo_label_received[max_pair_keys[1]]}
        return clusters_centers_dict

    def update_distance_of_all_clients(self, L2_of_all_clients, clusters_centers_dict):
        L2_temp = {}
        for pair in L2_of_all_clients.keys():
            id1, id2 = pair
            if (id1 in clusters_centers_dict) ^ (id2 in clusters_centers_dict):
                L2_temp[pair] = L2_of_all_clients[pair]
            #if id1 in clusters_centers_dict:
            #    if id2 not in clusters_centers_dict:
            #        L2_temp[pair] = L2_of_all_clients[pair]
            #if id2 in clusters_centers_dict:
            #    if id1 not in clusters_centers_dict:
            #        L2_temp[pair] = L2_of_all_clients[pair]

        return L2_temp


    def get_l2_of_non_centers(self,L2_of_all_clients,clusters_centers_dict):
        all_vals = {}
        for pair, l2 in L2_of_all_clients.items():
            if pair[0] in clusters_centers_dict:
                which_of_the_two = pair[1]
            elif pair[1] in clusters_centers_dict:
                which_of_the_two = pair[0]
            if which_of_the_two not in all_vals:
                all_vals[which_of_the_two] = []
            all_vals[which_of_the_two].append(l2)

        sum_of_vals = {}
        for k,v in all_vals.items():
            sum_of_vals[k]=sum(v)
        max_key = max(sum_of_vals, key=sum_of_vals.get)

        return max_key

    def complete_clusters_centers_and_L2_of_all_clients(self,clusters_centers_dict):
        if isinstance(experiment_config.num_clusters,str):
            cluster_counter = 3
        else:
            cluster_counter =  experiment_config.num_clusters -2

        while cluster_counter > 0:
            distance_of_all_clients = self.get_distance_dict()
            distance_of_all_clients = self.update_distance_of_all_clients(distance_of_all_clients, clusters_centers_dict)
            new_center = self.get_l2_of_non_centers(distance_of_all_clients,clusters_centers_dict)
            clusters_centers_dict[new_center] = self.pseudo_label_received[new_center]
            cluster_counter = cluster_counter-1
        distance_of_all_clients = self.get_distance_dict()
        distance_of_all_clients = self.update_distance_of_all_clients(distance_of_all_clients, clusters_centers_dict)
        return distance_of_all_clients,clusters_centers_dict

    def get_l2_of_non_center_to_center(self,L2_from_center_clients,clusters_centers_dict):
        ans ={}
        for pair, l2 in L2_from_center_clients.items():
            if pair[0] in clusters_centers_dict:
                not_center = pair[1]
                center = pair[0]
            elif pair[1] in clusters_centers_dict:
                not_center = pair[0]
                center = pair[1]
            if not_center not in ans:
                ans[not_center] = {}
            ans[not_center][center] = l2
        return ans

    def get_non_center_to_which_center_dict(self,l2_of_non_center_to_center):
        one_to_one_dict = {}
        for none_center,dict_ in l2_of_non_center_to_center.items():
            one_to_one_dict[none_center] =  min(dict_, key=dict_.get)

        dict_ = one_to_one_dict
        ans = {}
        for key, value in dict_.items():
            # Add the key to the list of the corresponding value in new_dict
            if value not in ans:
                ans[value] = []  # Initialize a list for the value if not present
            ans[value].append(key)

        return ans


    def prep_clusters(self,input_):
        ans = {}
        counter = 0
        for k, list_of_other in input_.items():
            ans[counter] = [k]
            for other_ in list_of_other:
                ans[counter].append(other_)
            counter = counter + 1
        return ans

    def get_clusters_centers_dict(self):
        L2_of_all_clients = self.get_distance_dict()
        clusters_centers_dict = self.initiate_clusters_centers_dict(L2_of_all_clients)
        L2_from_center_clients,clusters_centers_dict = self.complete_clusters_centers_and_L2_of_all_clients(clusters_centers_dict)
        L2_of_non_centers = self.get_l2_of_non_center_to_center(L2_from_center_clients,clusters_centers_dict)

        non_center_to_which_center_dict = self.get_non_center_to_which_center_dict(L2_of_non_centers)
        ans = self.prep_clusters(non_center_to_which_center_dict)
        centers_to_add = self.get_centers_to_add(clusters_centers_dict,ans)
        temp_ans = {}
        if len(centers_to_add)>0:
            for cluster_id in range(max(ans.keys())+1,max(ans.keys())+1+len(centers_to_add)):
                index = cluster_id-(max(ans.keys())+1)
                temp_ans[cluster_id]=centers_to_add[index]
        for k,v in temp_ans.items():
            ans[k]=[v]



        return ans
    def get_centers_to_add(self,clusters_centers_dict,ans):
        centers_to_add =[]
        for center_id in clusters_centers_dict.keys():
            center_to_add = self.center_id_not_in_ans(ans, center_id)
            if center_to_add is not None:
                centers_to_add.append(center_to_add)
        return centers_to_add
    def center_id_not_in_ans(self,ans,center_id):
        for list_of_id in ans.values():
            if center_id in list_of_id:
                return
        return center_id



    def manual_grouping(self):
        clusters_client_id_dict = {}
        if isinstance(experiment_config.num_clusters,int):
            num_clusters = experiment_config.num_clusters
        else:
            num_clusters = experiment_config.number_of_optimal_clusters

        if num_clusters == 1:
            clusters_client_id_dict[0]=self.clients_ids
        else:
            clusters_client_id_dict =  self.get_clusters_centers_dict()
        return clusters_client_id_dict







    def get_pseudo_labels_input_per_cluster(self,t):
        # Stack the pseudo labels tensors into a single tensor
        mean_per_cluster = {}

        flag = False
        clusters_client_id_dict = None
        if experiment_config.num_clusters == "Optimal":
            clusters_client_id_dict = experiment_config.known_clusters
            flag = True
        if experiment_config.num_clusters == 1:
            clusters_client_id_dict={0:self.clients_ids}
            #clusters_client_id_dict = experiment_config.known_clusters
            flag = True


        if (experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy or experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_L2) and not flag:
            if t == 0:
                clusters_client_id_dict = self.greedy_elimination()
            else:
                clusters_client_id_dict = self.clusters_client_id_dict_per_iter[0]


        if experiment_config.cluster_technique == ClusterTechnique.kmeans and not flag:
            clusters_client_id_dict = self.k_means_grouping()

        if (experiment_config.cluster_technique == ClusterTechnique.manual_L2 or experiment_config.cluster_technique == ClusterTechnique.manual_cross_entropy) and not flag:
            clusters_client_id_dict = self.manual_grouping()

        if experiment_config.cluster_technique == ClusterTechnique.manual_single_iter  and not flag:
            if t == 0:
                clusters_client_id_dict = self.manual_grouping()
            else:
                clusters_client_id_dict = self.clusters_client_id_dict_per_iter[0]


        cluster_mean_pseudo_labels_dict = self.get_cluster_mean_pseudo_labels_dict(clusters_client_id_dict)

        #if experiment_config.num_clusters>1:
        for cluster_id, pseudo_labels in cluster_mean_pseudo_labels_dict.items():
            pseudo_labels_list = list(pseudo_labels)
            if experiment_config.server_input_tech ==  ServerInputTech.mean:
                stacked_labels = torch.stack(pseudo_labels_list)
                # Average the pseudo labels across clients
                average_pseudo_labels = torch.mean(stacked_labels, dim=0)
                mean_per_cluster[cluster_id] = average_pseudo_labels
            if experiment_config.server_input_tech ==  ServerInputTech.median:
                stacked_labels = torch.stack(pseudo_labels_list)  # [num_clients, ...]
                # elementwise median across clients
                median_pseudo_labels, _ = torch.median(stacked_labels, dim=0)
                mean_per_cluster[cluster_id] = median_pseudo_labels

            if experiment_config.server_input_tech ==  ServerInputTech.max:
                mean_per_cluster[cluster_id] = self.select_confident_pseudo_labels(pseudo_labels_list)
        return mean_per_cluster, clusters_client_id_dict

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
        # experiment_config.batch_size
        global_data_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False)

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

    def __str__(self):
        return "server"

    def centroids_are_empty(self):
        for prev_cent in self.previous_centroids_dict.values():
            if prev_cent is None:
                return True
        else:
            return False

    def get_cluster_of_client(self,client_id,t):
        if experiment_config.cluster_technique == ClusterTechnique.manual_single_iter:
            for cluster_id, clients_id_list in self.clusters_client_id_dict_per_iter[0].items():
                if client_id in clients_id_list:
                    return cluster_id
        else:
            for cluster_id, clients_id_list in self.clusters_client_id_dict_per_iter[t].items():
                if client_id in clients_id_list:
                    return cluster_id
            print()

    def init_models_measures(self):
        num_clusters = experiment_config.num_clusters
        for cluster_id in range(num_clusters):
            self.previous_centroids_dict[cluster_id] = None
            self.accuracy_server_test_1[cluster_id] = {}
            self.accuracy_global_data_1[cluster_id] = {}
            self.multi_model_dict[cluster_id] = get_server_model()
            self.multi_model_dict[cluster_id].apply(self.initialize_weights)

    def get_distance_per_client(self,distance_dict):
        ans = {}
        for pair, dist in distance_dict.items():
            first_id = pair[0]
            second_id = pair[1]
            if first_id not in ans:
                ans[first_id] = {}
            if second_id not in ans:
                ans[second_id] = {}
            ans[first_id][second_id] = dist
            ans[second_id][first_id] = dist

        return ans




    def greedy_elimination(self):
        distance_dict = self.get_distance_dict()
        distance_per_client = self.get_distance_per_client(distance_dict)
        epsilon_ = self.calc_epsilon(distance_per_client)
        clusters_client_id_dict = self.greedy_elimination_t0(epsilon_,distance_per_client)
        experiment_config.num_clusters = len(clusters_client_id_dict)
        self.init_models_measures()
        return clusters_client_id_dict
            #experiment_config.num_clusters, clusters_client_id_dict = self.greedy_elimination_t_larger(epsilon_=None,
            #                                                                                           k=experiment_config.num_clusters)

    def get_distance_dict(self):
        pairs = list(combinations(self.clients_ids, 2))
        distance_dict = {}
        for pair in pairs:
            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_L2 or  experiment_config.cluster_technique == ClusterTechnique.manual_L2:
                distance_dict[pair] = self.calc_L2(pair)
            if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy or experiment_config.cluster_technique == ClusterTechnique.manual_cross_entropy:
                distance_dict[pair] = self.calc_cross_entropy(pair)
        return distance_dict

    def calc_cross_entropy(self, pair):
        first_pl = self.pseudo_label_received[pair[0]]
        second_pl = self.pseudo_label_received[pair[1]]
        return Server.calc_cross_entropy_given_pl(first_pl,second_pl)


    @staticmethod
    def calc_cross_entropy_given_pl(first_pl,second_pl):
        loss1 = -(first_pl * torch.log(second_pl)).sum(dim=1).mean()
        loss2 = -(second_pl * torch.log(first_pl.clamp(min=1e-9))).sum(dim=1).mean()
        loss = 0.5 * (loss1 + loss2)
        return loss.item()


    def get_pseudo_label_in_cluster(self,clusters_client_id_dict):
        pseudo_labels_in_cluster = {}
        for cluster_id, clients in clusters_client_id_dict.items():
            pseudo_labels_in_cluster[cluster_id] = []
            for client_id in clients:
                pl = self.pseudo_label_received[client_id]
                pseudo_labels_in_cluster[cluster_id].append(pl)
        return pseudo_labels_in_cluster

    def calc_epsilon(self,distance_per_client):
        epsilon_ = 0.1
        while True:
            amount_clusters = len(self.greedy_elimination_t0(epsilon_,copy.deepcopy(distance_per_client)))
            if amount_clusters<=experiment_config.number_of_optimal_clusters+experiment_config.cluster_addition:
                break
            else:
                epsilon_ = epsilon_+0.1
        return epsilon_

        #clusters_client_id_dict = experiment_config.known_clusters
        #pseudo_labels_in_cluster = self.get_pseudo_label_in_cluster(clusters_client_id_dict)

        #center_of_cluster = {}
        #for cluster_id,list_of_pseudo_labels in pseudo_labels_in_cluster.items():
        #    center_of_cluster[cluster_id] = torch.stack(list_of_pseudo_labels).mean(dim=0)

        #if experiment_config.cluster_technique == ClusterTechnique.greedy_elimination_cross_entropy:
        #    distance_dict = self.compute_distances(center_of_cluster,Server.calc_cross_entropy_given_pl)
        #else:
        #    distance_dict = self.compute_distances(center_of_cluster, Server.calc_L2_given_pls)

        #min_distance = min(distance_dict.values())
        #return min_distance*experiment_config.epsilon#(4.2/5)



    def compute_distances(self,id_label_dict, distance_function):
        """
        Given a dictionary of {id: pseudo_label}, compute the pairwise distances.

        Args:
        id_label_dict (dict): Dictionary where keys are IDs and values are pseudo labels.
        distance_function (function): Function that computes distance between two pseudo labels.

        Returns:
        dict: Dictionary where keys are (id1, id2) tuples and values are distances.
        """
        distance_dict = {}
        ids = list(id_label_dict.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                distance = distance_function(id_label_dict[id1], id_label_dict[id2])
                distance_dict[(id1, id2)] = distance

        return distance_dict

    def filter_far_clients(self,distance_per_client,epsilon):
        for client, distance_dict in distance_per_client.items():
            far_clients = []
            for other_client, distance in distance_dict.items():
                if distance > epsilon:
                    far_clients.append(other_client)
            for other_client in far_clients:
                del distance_dict[other_client]

    def greedy_elimination_t0(self, epsilon_, distance_per_client):
        self.filter_far_clients(distance_per_client,epsilon_)#what is the distance between 10 and 11
        clusters_client_id_dict = {}
        counter = -1
        while len(distance_per_client)>0:
            counter = counter + 1

            max_client = max(distance_per_client, key=lambda k: len(distance_per_client[k]))
            others_to_remove = list(distance_per_client[max_client].keys())
            lst = copy.deepcopy(others_to_remove)
            lst.append(max_client)
            clusters_client_id_dict[counter] =lst
            for other in clusters_client_id_dict[counter]:
                for other_in_distance_per_client in distance_per_client.values():
                    if other in other_in_distance_per_client:
                        del other_in_distance_per_client[other]
                del distance_per_client[other]
        return clusters_client_id_dict

import copy
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ======= Helper: apply masks (elementwise) =======
def _masked_copy_(dst_param: torch.nn.Parameter, src_tensor: torch.Tensor, mask: torch.Tensor):
    # Only copy entries where mask == 1 (shared part)
    with torch.no_grad():
        dst_param.data[mask.bool()] = src_tensor.data[mask.bool()]

def _masked_zero_grad_(param: torch.nn.Parameter, mask_personal: torch.Tensor, keep_shared: bool):
    """
    If keep_shared=True  -> zero grads on personalized entries (so shared gets updated)
    If keep_shared=False -> zero grads on shared entries (so personalized gets updated)
    """
    if param.grad is None:
        return
    m = mask_personal.bool()
    if keep_shared:
        # zero personalized grads
        param.grad.data[m] = 0
    else:
        # zero shared grads (= complement)
        param.grad.data[~m] = 0

def _all_params(model: nn.Module) :
    return [p for p in model.parameters() if p.requires_grad]

# ======= Client: FedSelect =======
class Client_FedSelect(Client):
    """
    FedSelect-style client with compatibility shims:
      - Produces `weights_to_send` (shared-only state dict; personalized entries zeroed)
      - Consumes `weights_received` by applying only to shared positions
      - Exposes `.iterate(t)` for your existing loop
    """

    # ---------- helpers ----------
    @staticmethod
    def _all_params(model):
        return [p for p in model.parameters() if p.requires_grad]

    @staticmethod
    def _masked_zero_grad_(param, mask_personal, keep_shared: bool):
        if param.grad is None:
            return
        m = mask_personal.bool()
        if keep_shared:
            param.grad.data[m] = 0
        else:
            param.grad.data[~m] = 0

    # ---------- init ----------
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)

        # FedSelect masks keyed by param name (buffers are not masked)
        self._init_fedselect_masks()

        # FedSelect hyperparams (fall back to your experiment_config)
        self.fs_select_every      = getattr(experiment_config, "fedselect_select_every", 1)
        self.fs_grow_frac         = getattr(experiment_config, "fedselect_grow_frac", 0.05)
        self.fs_warmup_rounds     = getattr(experiment_config, "fedselect_warmup_rounds", 0)
        self.local_epochs_shared   = getattr(experiment_config, "fedselect_epochs_shared", 1)
        self.local_epochs_personal = getattr(experiment_config, "fedselect_epochs_personal", 1)
        self.lr_shared     = getattr(experiment_config, "fedselect_lr_shared",     experiment_config.learning_rate_train_c)
        self.lr_personal   = getattr(experiment_config, "fedselect_lr_personal",   experiment_config.learning_rate_fine_tune_c)
        self.batch_size    = getattr(experiment_config, "fedselect_batch_size",    experiment_config.batch_size)

        # --- compatibility fields expected by your loop ---
        self.weights_to_send = None
        self._weights_received = None  # private; use property below

        # cache for upload
        self.last_shared_state = {}

    # Provide the old attribute with an auto-apply on set
    @property
    def weights_received(self):
        return self._weights_received

    @weights_received.setter
    def weights_received(self, state):
        self._weights_received = state
        if state is not None:
            # server sends aggregated SHARED slice; write only shared positions
            self.apply_aggregated_shared(state)

    def _init_fedselect_masks(self):
        self.fs_masks_by_name = {}
        for name, p in self.model.named_parameters():
            self.fs_masks_by_name[name] = torch.zeros_like(p.data, dtype=torch.bool, device=p.device)

    def _importance_scores_by_name(self):
        self.model.zero_grad(set_to_none=True)
        tmp_loader = DataLoader(self.local_data, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        max_batches = getattr(experiment_config, "fedselect_score_batches", 1)
        seen = 0
        for xb, yb in tmp_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = self.model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            seen += 1
            if seen >= max_batches:
                break

        scores = {}
        for name, p in self.model.named_parameters():
            scores[name] = torch.zeros_like(p.data) if p.grad is None else p.grad.detach().abs()
        return scores

    def _grow_personal_mask(self, t_round: int):
        if t_round < self.fs_warmup_rounds: return
        if self.fs_grow_frac <= 0: return
        if (t_round % self.fs_select_every) != 0: return

        scores_by_name = self._importance_scores_by_name()
        for name, p in self.model.named_parameters():
            mask = self.fs_masks_by_name[name]
            with torch.no_grad():
                shared_positions = (~mask).flatten()
                if shared_positions.sum() == 0:
                    continue
                s = scores_by_name[name].flatten()[shared_positions]
                k = max(1, int(self.fs_grow_frac * s.numel()))
                _, topk_idx = torch.topk(s, k, largest=True, sorted=False)
                full_idx = shared_positions.nonzero(as_tuple=False).squeeze(1)[topk_idx]
                mf = mask.flatten()
                mf[full_idx] = True
                self.fs_masks_by_name[name] = mf.view_as(mask)

    def _local_optimize(self, data_loader, epochs: int, keep_shared: bool, lr: float):
        self.model.train()
        opt = torch.optim.Adam(self._all_params(self.model), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                for name, p in self.model.named_parameters():
                    m = self.fs_masks_by_name[name]
                    self._masked_zero_grad_(p, m, keep_shared=keep_shared)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()

    def iteration_context(self, t: int):
        self.current_iteration = t

        # grow per schedule
        self._grow_personal_mask(t)

        # shared phase
        train_loader = DataLoader(self.local_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self._local_optimize(train_loader, self.local_epochs_shared, keep_shared=True, lr=self.lr_shared)

        # personal phase
        self._local_optimize(train_loader, self.local_epochs_personal, keep_shared=False, lr=self.lr_personal)

        # package shared-only state (params zeroed on personalized entries) + full buffers
        payload = {}
        for name, p in self.model.named_parameters():
            m = self.fs_masks_by_name[name]
            if m.shape != p.shape:
                raise RuntimeError(f"[FedSelect] Mask shape mismatch for {name}: {m.shape} vs {p.shape}")
            shared_tensor = p.detach().clone()
            shared_tensor[m] = 0
            payload[name] = shared_tensor
        for bname, buf in self.model.named_buffers():
            payload[bname] = buf.detach().clone()

        self.last_shared_state = payload

        # keep your metrics
        self.accuracy_per_client_1[t]   = self.evaluate_accuracy_single(self.local_test_set, k=1)
        self.accuracy_per_client_5[t]   = self.evaluate_accuracy(self.local_test_set, k=5)
        self.accuracy_per_client_10[t]  = self.evaluate_accuracy(self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)

    # --- compatibility: old loop calls c.iterate(t) and reads c.weights_to_send ---
    def iterate(self, t: int):
        self.iteration_context(t)
        self.weights_to_send = self.last_shared_state  # exactly what your loop expects

    # FedSelect uploads params, not pseudo labels
    @property
    def fedselect_payload(self):
        return self.last_shared_state

    def apply_aggregated_shared(self, aggregated_state):
        with torch.no_grad():
            cur = self.model.state_dict()
            # parameters: write only shared positions
            for name, p in self.model.named_parameters():
                if name not in aggregated_state:
                    continue
                m = self.fs_masks_by_name[name]
                src = aggregated_state[name].to(p.device)
                p.data[~m] = src.data[~m]
            # buffers: overwrite fully if present
            for bname, buf in self.model.named_buffers():
                if bname in aggregated_state:
                    buf.copy_(aggregated_state[bname].to(buf.device))

# ======= Server: FedSelect =======
class Server_FedSelect(Server):
    """
    FedSelect server that aggregates *shared-only* client states (personalized entries are zeroed on client).
    Keeps original FedAvg I/O:
      - fill `received_weights[client_id]` before `.iterate(t)`
      - after `.iterate(t)`, read `weights_to_send[client_id]`
    """
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict, clients: List[Client_FedSelect]):
        super().__init__(id_, global_data, test_data, clients_ids, clients_test_data_dict)
        self.clients = clients

        # mirror client architecture for logging/eval of shared part
        self.global_model = copy.deepcopy(clients[0].model).to(device)

        # --- compatibility dicts expected by your loop ---
        self.received_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        self.weights_to_send: Dict[str, Dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def _fedavg_on_shared(self, shared_list: List[Dict[str, torch.Tensor]]) :
        """
        Robust FedAvg over *floating* tensors that share identical shapes across all clients.
        - Non-floating tensors (e.g., int buffers) are copied from the first client.
        - Keys missing on any client are excluded from averaging.
        - Shape mismatches fall back to copying from the first client (no averaging).
        - Averages are computed in fp32 for numerical stability, then cast back.
        """
        agg: Dict[str, torch.Tensor] = {}
        if not shared_list:
            return agg

        # Only average keys present in *all* client payloads
        common_keys = set(shared_list[0].keys())
        for sd in shared_list[1:]:
            common_keys &= set(sd.keys())
        if not common_keys:
            return agg

        # If we have a reference model, prefer its floating params for averaging
        ref_state = getattr(self, "global_model", None).state_dict() if hasattr(self, "global_model") else None

        for k in sorted(common_keys):
            tensors = [sd[k] for sd in shared_list]
            # Ensure all are tensors
            if any(not torch.is_tensor(t) for t in tensors):
                agg[k] = tensors[0].to(device)
                continue

            # Shapes identical?
            shapes = {tuple(t.shape) for t in tensors}
            if len(shapes) != 1:
                # fallback: just copy first (don’t try to stack/mean)
                agg[k] = tensors[0].to(device)
                continue

            # Floating point? Then we can average. Otherwise copy first.
            if all(t.is_floating_point() for t in tensors):
                # Promote to fp32 for mean, then cast back to original dtype
                base_dtype = tensors[0].dtype
                stacked = torch.stack([t.to(device, dtype=torch.float32) for t in tensors], dim=0)
                mean_fp32 = stacked.mean(dim=0)
                out = mean_fp32.to(dtype=base_dtype)

                # Optional: if we have a ref_state and its dtype differs, match it
                if ref_state is not None and k in ref_state and ref_state[k].dtype != out.dtype:
                    out = out.to(ref_state[k].dtype)
                agg[k] = out
            else:
                # integer/bool buffers (e.g., num_batches_tracked) – don’t average
                agg[k] = tensors[0].to(device)

        return agg

    def iterate(self, t: int):
        """
        Old-style server step:
          - aggregate from self.received_weights (dict client_id -> state_dict)
          - update self.global_model
          - set self.weights_to_send[client_id] = aggregated_shared for all clients
        """
        self.current_iteration = t

        # 1) collect client payloads
        shared_list = []
        for cid in self.clients_ids:
            state = self.received_weights.get(cid, None)
            if state is None:
                # If a client didn't send this round, skip it
                continue
            # make sure tensors are on CPU/GPU consistently
            shared_list.append(state)

        if len(shared_list) == 0:
            # nothing to aggregate this round
            self.weights_to_send = {cid: {} for cid in self.clients_ids}
            return

        # 2) aggregate
        aggregated = self._fedavg_on_shared(shared_list)

        # 3) load into server's global model (for optional eval)
        gm = self.global_model.state_dict()
        for name in gm.keys():
            if name in aggregated:
                gm[name] = aggregated[name].to(gm[name].device)
        self.global_model.load_state_dict(gm)

        # 4) broadcast back in the same FedAvg-compatible way
        self.weights_to_send = {cid: aggregated for cid in self.clients_ids}

        # 5) (Optional) log shared-model accuracy like your server does
        self.accuracy_server_test_1.setdefault(0, {})[t] = self.evaluate_accuracy_single(
            self.test_global_data, model=self.global_model, k=1, cluster_id=0)
        self.accuracy_global_data_1.setdefault(0, {})[t] = self.evaluate_accuracy_single(
            self.global_data, model=self.global_model, k=1, cluster_id=0)



# ---- pFedHN Client & Server with round hooks ----
# Assumes you already have: torch, torch.nn as nn, torch.utils.data.DataLoader,
# experiment_config, device, and your base Client/Server classes in scope.

from typing import Dict, List, Optional
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------- small helper: robust averaging restricted to a reference state ----------
@torch.no_grad()
def _robust_fedavg_state_dict(state_dicts: List[Dict[str, torch.Tensor]],
                              ref_state: Dict[str, torch.Tensor],
                              device: torch.device) :
    """
    Average only keys present in ref_state. Silently skips missing keys in inputs.
    Casts to ref dtype/device. Returns a flat state_dict (no in-place on ref).
    """
    agg: Dict[str, torch.Tensor] = {}
    for k, ref_t in ref_state.items():
        bucket = []
        for sd in state_dicts:
            v = sd.get(k, None)
            if v is None:
                continue
            bucket.append(v.to(device=device, dtype=ref_t.dtype))
        if len(bucket) == 0:
            continue
        stacked = torch.stack(bucket, dim=0)
        agg[k] = stacked.mean(dim=0)
    return agg


# =========================
# Client (pFedHN)
# =========================
class Client_pFedHN(Client):
    """
    pFedHN-style client: keeps a local hypernetwork (HN) which generates personalized target nets.
    We upload ONLY the HN parameters (not the generated target params).
    """
    HN_PREFIXES = ("hypernet.", "hn.",)  # adapt to your model's HN parameter name prefixes

    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)
        self.batch_size = getattr(experiment_config, "pfedhn_batch_size", experiment_config.batch_size)
        self.local_epochs = getattr(experiment_config, "pfedhn_local_epochs", 1)
        self.lr = getattr(experiment_config, "pfedhn_lr", experiment_config.learning_rate_train_c)

        # caches for upload / round bookkeeping
        self._last_hn_state: Dict[str, torch.Tensor] = {}
        self._last_agg_hn: Optional[Dict[str, torch.Tensor]] = None

    # ----------------- hooks you asked for -----------------
    def generate_weights_for(self, client_id: Optional[str] = None):
        """
        If your model exposes explicit hypernetwork weight generation for a given id/embedding,
        call it here. Otherwise, this is a safe no-op. We still set an id if your model expects it.
        """
        if hasattr(self.model, "set_client_id"):
            try:
                self.model.set_client_id(client_id if client_id is not None else self.id_)
            except Exception:
                pass

        # Optional explicit generation calls if present in your model:
        if hasattr(self.model, "generate_weights_for"):
            try:
                return self.model.generate_weights_for(client_id if client_id is not None else self.id_)
            except Exception:
                return None
        if hasattr(self.model, "generate"):
            try:
                return self.model.generate()
            except Exception:
                return None
        return None

    @torch.no_grad()
    def apply_client_update(self, aggregated_hn: Dict[str, torch.Tensor]):
        """
        Apply a *full* hypernetwork state update coming from the server (overwrite HN slice).
        """
        self.apply_hn_shared(aggregated_hn)
        self._last_agg_hn = aggregated_hn

    @torch.no_grad()
    def apply_client_delta(self, hn_delta: Dict[str, torch.Tensor]):
        """
        Apply an *incremental* delta to the hypernetwork parameters (HN slice only).
        Useful if the server sends a delta instead of a full state.
        """
        cur = self.model.state_dict()
        changed = False
        for k, dv in hn_delta.items():
            if k not in cur:
                continue
            cur[k] = (cur[k].to(dv.device, dv.dtype) + dv).to(cur[k].device, cur[k].dtype)
            changed = True
        if changed:
            self.model.load_state_dict(cur, strict=False)

    def after_round(self, t: int):
        """
        Any per-round client cleanup or logging hook.
        Right now we just keep it as a convenient place to extend later.
        """
        # Example: regenerate weights bound to this client_id for the next round (if relevant)
        self.generate_weights_for(self.id_)

    # ----------------- internal helpers -----------------
    def _hypernet_state(self) :
        full = self.model.state_dict()
        def is_hn_key(k: str) :
            return any(k.startswith(pref) for pref in self.HN_PREFIXES)
        return {k: v.detach().clone() for k, v in full.items() if is_hn_key(k)}

    def _local_train_hn(self):
        self.model.train()
        hn_params = [p for n, p in self.model.named_parameters()
                     if any(n.startswith(pref) for pref in self.HN_PREFIXES)]
        if len(hn_params) == 0:
            # If prefixes are mismatched, fall back to training all params (won't crash)
            hn_params = list(self.model.parameters())

        opt = torch.optim.Adam(hn_params, lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(self.local_data, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # (Optional) ensure the HN is "conditioned" for this client before training
        self.generate_weights_for(self.id_)

        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)  # forward should internally use HN → target weights
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(hn_params, max_norm=1.0)
                opt.step()

    # ----------------- pFedHN round entrypoint -----------------
    def iteration_context(self, t: int):
        self.current_iteration = t
        self._local_train_hn()

        # Upload-only the hypernetwork slice
        self._last_hn_state = self._hypernet_state()

        # (Optional) local eval logging
        self.accuracy_per_client_1[t] = self.evaluate_accuracy_single(self.local_test_set, k=1)
        self.accuracy_per_client_5[t] = self.evaluate_accuracy(self.local_test_set, k=5)
        self.accuracy_per_client_10[t] = self.evaluate_accuracy(self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)

    @property
    def pfedhn_payload(self) :
        return self._last_hn_state

    @torch.no_grad()
    def apply_hn_shared(self, aggregated_hn: Dict[str, torch.Tensor]):
        """
        Server returns aggregated *HN* params → write them into the client's model.
        (Only keys that belong to the hypernetwork slice are touched.)
        """
        cur = self.model.state_dict()
        for k, v in aggregated_hn.items():
            if k in cur:
                cur[k] = v.to(cur[k].device).to(cur[k].dtype)
        self.model.load_state_dict(cur, strict=False)


# =========================
# Server (pFedHN)
# =========================
class Server_pFedHN(Server):
    """
    Server for pFedHN: aggregates ONLY hypernetwork (HN) parameters across clients.
    Provides hooks:
      - generate_weights_for(client_id)
      - apply_client_update(client_id, hn_state)
      - apply_client_delta(client_id, hn_delta)
      - after_round(t)
    """
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict,
                 clients: List[Client_pFedHN]):
        super().__init__(id_, global_data, test_data, clients_ids, clients_test_data_dict)
        self.clients = clients

        # Build a server-side "global HN" by cloning client[0]'s model
        self.global_model = copy.deepcopy(clients[0].model).to(device)

        # Cache HN keys once (based on client 0)
        self.hn_keys = [k for k in self.global_model.state_dict().keys()
                        if any(k.startswith(pref) for pref in clients[0].HN_PREFIXES)]

        # Per-round inbox for received HN (full state or deltas, if you want)
        self._inbox_hn_states: Dict[str, Dict[str, torch.Tensor]] = {}

    # ----------------- hooks you asked for -----------------
    @torch.no_grad()
    def generate_weights_for(self, client_id: Optional[str] = None):
        """
        If your server needs to instantiate target weights (e.g., for centralized eval),
        call the model’s generator here, otherwise this is a safe no-op.
        """
        if hasattr(self.global_model, "set_client_id"):
            try:
                self.global_model.set_client_id(client_id)
            except Exception:
                pass
        if hasattr(self.global_model, "generate_weights_for"):
            try:
                return self.global_model.generate_weights_for(client_id)
            except Exception:
                return None
        if hasattr(self.global_model, "generate"):
            try:
                return self.global_model.generate()
            except Exception:
                return None
        return None

    @torch.no_grad()
    def apply_client_update(self, client_id: str, hn_state: Dict[str, torch.Tensor]):
        """
        Receive a *full* HN slice from a client during the round.
        (Stored and aggregated at the end of the round.)
        """
        # Filter to HN keys only
        filtered = {k: v.detach().clone() for k, v in hn_state.items() if k in self.hn_keys}
        self._inbox_hn_states[client_id] = filtered

    @torch.no_grad()
    def apply_client_delta(self, client_id: str, hn_delta: Dict[str, torch.Tensor]):
        """
        Add a *delta* to the server's global HN immediately (rarely used; provided for completeness).
        """
        cur = self.global_model.state_dict()
        changed = False
        for k, dv in hn_delta.items():
            if k not in self.hn_keys or k not in cur:
                continue
            cur[k] = (cur[k].to(dv.device, dv.dtype) + dv).to(cur[k].device, cur[k].dtype)
            changed = True
        if changed:
            self.global_model.load_state_dict(cur, strict=False)

    @torch.no_grad()
    def after_round(self, t: int):
        """
        Aggregate the per-round inbox, update server HN, and broadcast to clients.
        """
        if len(self._inbox_hn_states) == 0:
            # nothing received this round
            return

        ref = self.global_model.state_dict()
        # pack into list for averaging
        hn_list = list(self._inbox_hn_states.values())
        aggregated_hn = _robust_fedavg_state_dict(hn_list, ref_state=ref, device=device)

        # Write aggregated HN into server model
        gm = self.global_model.state_dict()
        for k, v in aggregated_hn.items():
            gm[k] = v.to(gm[k].device).to(gm[k].dtype)
        self.global_model.load_state_dict(gm, strict=False)

        # Broadcast aggregated HN to clients
        for c in self.clients:
            c.apply_client_update(aggregated_hn)

        # Optional: centralized eval with global HN
        self.accuracy_server_test_1.setdefault(0, {})[t] = self.evaluate_accuracy_single(
            self.test_global_data, model=self.global_model, k=1, cluster_id=0
        )
        self.accuracy_global_data_1.setdefault(0, {})[t] = self.evaluate_accuracy_single(
            self.global_data, model=self.global_model, k=1, cluster_id=0
        )

        # Clear inbox
        self._inbox_hn_states.clear()

    # ----------------- legacy-style helpers (kept for compatibility) -----------------
    def _receive_hn_states(self) :
        # If you still want the older pull-style, use the client payloads:
        return [c.pfedhn_payload for c in self.clients]

    def iteration_context(self, t: int):
        """
        If you prefer the legacy single-call style (pull → aggregate → push),
        keep this. It also calls after_round(t) so both styles work.
        """
        self.current_iteration = t

        # Pull client payloads:
        for c in self.clients:
            self.apply_client_update(c.id_, c.pfedhn_payload)

        # End-of-round aggregate + broadcast:
        self.after_round(t)


# ---- in your model code, expose mid-layer features ----
# example: modify your client nets so forward can return (logits, feat)
# def forward(self, x, return_feat=False): ...
#   if return_feat: return logits, feat
#   else: return logits

import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader






import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Client_Ditto(Client):
    """
    Ditto client:
      - self.model:          global-path model (sent to server; FedAvg)
      - self.personal_model: local personalized model (kept on client)
      - self.weights_received: set by server after aggregation (expected each round)
      - self.weights_to_send: what we send to the server (global path)
    """

    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data, lam_ditto=1.0):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)
        self.personal_model = copy.deepcopy(self.model)
        self.lam_ditto = float(lam_ditto)

        self.weights_received = None
        self.weights_to_send  = None

        # Optional separate epoch knobs; will fallback to your existing var if absent.
        self.epochs_global   = getattr(experiment_config, "epochs_global",
                                getattr(experiment_config, "epochs_num_input_fine_tune_clients", 1))
        self.epochs_personal = getattr(experiment_config, "epochs_personal",
                                getattr(experiment_config, "epochs_num_input_fine_tune_clients", 1))

        self.lr_c = experiment_config.learning_rate_fine_tune_c  # reuse your LR

    # ---------- utilities ----------
    @torch.no_grad()
    def _load_state(self, model, state_dict):
        model.load_state_dict(state_dict, strict=True)

    def _eval_on(self, model, eval_fn, *args, **kwargs):
        """Temporarily swap self.model to reuse your existing eval funcs unchanged."""
        old = self.model
        try:
            self.model = model
            return eval_fn(*args, **kwargs)
        finally:
            self.model = old

    def _proximal_loss_L2_mean(self, model, anchor_state):
        """
        Mean L2 distance (not sum) between params and anchor.
        Using mean makes lambda human-scaled (e.g., 0.05–2).
        """
        anchor = {k: v.to(next(model.parameters()).device) for k, v in anchor_state.items()}
        sq, n = 0.0, 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            d = p - anchor[name]
            sq += torch.sum(d * d)
            n  += d.numel()
        return sq / max(n, 1)

    # ---------- training paths ----------
    def _global_update_fedavg(self):
        """Local supervised training on self.model; returns sd to send."""
        print(f"*** {self} global-path (FedAvg) update ***")
        loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size, shuffle=True)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr_c)

        for epoch in range(self.epochs_global):
            self.epoch_count += 1
            epoch_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += float(loss.item())
            print(f"[Global] Epoch {epoch+1}/{self.epochs_global}  Loss: {epoch_loss/len(loader):.4f}")

        return copy.deepcopy(self.model.state_dict())

    def _personalized_update_ditto(self, anchor_state):
        """Ditto objective on self.personal_model: CE + 0.5*λ*mean||θ-v||^2."""
        print(f"*** {self} personalized-path (Ditto) update, lambda={self.lam_ditto} ***")
        loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size, shuffle=True)
        self.personal_model.train()
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.personal_model.parameters(), lr=self.lr_c)

        # Freeze a copy of the anchor
        anchor = {k: v.detach().clone() for k, v in anchor_state.items()}

        for epoch in range(self.epochs_personal):
            epoch_obj = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = self.personal_model(x)
                ce = criterion(logits, y)
                prox = self._proximal_loss_L2_mean(self.personal_model, anchor)
                loss = ce + 0.5 * self.lam_ditto * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.personal_model.parameters(), max_norm=1.0)
                opt.step()
                epoch_obj += float(loss.item())
            print(f"[Personalized] Epoch {epoch+1}/{self.epochs_personal}  Obj: {epoch_obj/len(loader):.4f}")

    # ---------- round entry ----------
    def iterate(self, t):
        self.current_iteration = t

        # (A) Model init / loading
        if t == 0:
            if self.weights_received is not None:
                # Best case: server broadcasted a common w0
                self._load_state(self.model, self.weights_received)
                self.personal_model = copy.deepcopy(self.model)
                print(f"{self}: loaded broadcast w0")
            else:
                # Fallback (try to avoid by broadcasting w0 from server)
                self.model.apply(self.initialize_weights)
                self.personal_model = copy.deepcopy(self.model)
                print(f"{self}: WARNING—no broadcast w0; using local init")
        else:
            # Load aggregated global for this round
            self._load_state(self.model, self.weights_received)

        # Keep a PRE-UPDATE anchor in case no broadcast happened this round
        pre_update_anchor = copy.deepcopy(self.model.state_dict())

        # (B) Global-path update (what we send to server)
        self.weights_to_send = self._global_update_fedavg()

        # (C) Personalized Ditto update (kept local)
        # Prefer the received global; fallback to the pre-update anchor on round 0
        anchor = self.weights_received if self.weights_received is not None else pre_update_anchor
        self._personalized_update_ditto(anchor)

        # (D) Accounting & metrics
        total_size = sum(p.numel() * p.element_size() for p in self.weights_to_send.values())
        self.size_sent[t] = total_size / (1024 * 1024)

        # Evaluate with personalized model on local test
        self.accuracy_per_client_1[t]   = self._eval_on(self.personal_model, self.evaluate_accuracy_single, self.local_test_set, k=1)
        self.accuracy_per_client_5[t]   = self._eval_on(self.personal_model, self.evaluate_accuracy,        self.local_test_set, k=5)
        self.accuracy_per_client_10[t]  = self._eval_on(self.personal_model, self.evaluate_accuracy,        self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self._eval_on(self.personal_model, self.evaluate_accuracy,        self.local_test_set, k=100)

        # Optional: also log global-path accuracy on global test
        _ = self.evaluate_accuracy_single(self.test_global_data)
        return


class Client_NoFederatedLearning(Client):
    def __init__(self,id_, client_data, global_data,global_test_data,local_test_data,evaluate_every):
        Client.__init__(self,id_, client_data, global_data,global_test_data,local_test_data)
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
            if   epoch % self.evaluate_every == 0 and epoch!=0:
                self.accuracy_per_client_1[epoch] = self.evaluate_accuracy_single(self.local_test_set, k=1)
                self.accuracy_per_client_5[epoch] = self.evaluate_accuracy(self.local_test_set, k=5)
                self.accuracy_per_client_10[epoch] = self.evaluate_accuracy(self.local_test_set, k=10)
                self.accuracy_per_client_100[epoch] = self.evaluate_accuracy(self.local_test_set, k=100)

            result_to_print = epoch_loss / len(fine_tune_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {result_to_print:.4f}")
        #self.weights = self.model.state_dict()self.weights = self.model.state_dict()

        return  result_to_print

class Client_PseudoLabelsClusters_with_division(Client):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        Client.__init__(self,id_, client_data, global_data,global_test_data,local_test_data)



    def train(self,mean_pseudo_labels):

        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(self.global_data[self.current_iteration-1], batch_size=experiment_config.batch_size, shuffle=False, num_workers=0,
                                   drop_last=True)
        #server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False,
        #                           num_workers=0)
        #print(1)
        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam( self.model.parameters(), lr=experiment_config.learning_rate_train_c)
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_c,
        #                             weight_decay=1e-4)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train_client):
            #print(2)

            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                #print(batch_idx)

                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs =  self.model(inputs)
                # Check for NaN or Inf in outputs

                # Convert model outputs to log probabilities
                outputs_prob = F.log_softmax(outputs, dim=1)
                # Slice pseudo_targets to match the input batch size
                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)
                pseudo_targets = self.apply_temperature_to_probs(pseudo_targets)

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
                #pseudo_targets = F.softmax(pseudo_targets, dim=1)

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

        #self.weights =self.model.state_dict()
        return avg_loss


    def evaluate(self, model=None):
        if model is None:
            model = self.model
    #    print("*** Generating Pseudo-Labels with Probabilities ***")

        # Create a DataLoader for the global data
        global_data_loader = DataLoader(self.global_data[self.current_iteration], batch_size=experiment_config.batch_size, shuffle=False)

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

       #print(f"Shape of the 2D pseudo-label matrix: {all_probs.shape}")
        return all_probs


import numpy as np, copy, torch
from sklearn.cluster import KMeans
class Client_pFedCK(Client):
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)
        self.rnd_net = Random((self.seed + 1) * 17 + 13 + (id_ + 1) * 17)
        self.personalized_model,ttt = self.get_client_model()
        self.interactive_model,ttt  = self.get_client_model()
        self.initial_state = None  # for delta ω

    def set_models(self, personalized_state, interactive_state):
        self.personalized_model.load_state_dict(personalized_state)
        self.interactive_model.load_state_dict(interactive_state)

    def _pick_feature_module(self, model):
        """
        Best-effort choice of a mid/penultimate feature module to hook.
        Works for common backbones you use (AlexNet/VGG/MobileNetV2Server/*rnd*).
        """
        # Prefer explicit attributes if they exist
        for name in ["features", "backbone", "encoder", "stem"]:
            if hasattr(model, name) and isinstance(getattr(model, name), torch.nn.Module):
                return getattr(model, name)
        # Fallbacks: try to find a big Sequential with convs
        cand = None
        for m in model.modules():
            # first reasonably deep Sequential is a decent proxy
            if isinstance(m, torch.nn.Sequential) and len(list(m.children())) >= 3:
                cand = m
                break
        return cand if cand is not None else model  # ultimate fallback

    def _forward_with_feat(self, model, x):
        """
        Returns (logits, feat) without needing return_feat support.
        Feature is a flattened tensor from a mid/penultimate layer.
        """
        captured = {}
        layer = self._pick_feature_module(model)

        def hook(_, __, out):
            captured["feat"] = out

        h = layer.register_forward_hook(hook)
        try:
            logits = model(x)
        finally:
            h.remove()

        feat = captured.get("feat", None)
        if feat is None:
            # As a last resort, derive a 'feature' from logits (not ideal, but safe)
            feat = logits

        # Flatten spatial dims if present (e.g., N,C,H,W -> N,C)
        if feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        return logits, feat

    def train(self, t):
        self.current_iteration = t
        # snapshot interactive model BEFORE local training
        with torch.no_grad():
            self.initial_state = {k: v.detach().clone() for k, v in self.interactive_model.state_dict().items()}

        self.personalized_model.train()
        self.interactive_model.train()

        opt_phi  = torch.optim.Adam(self.personalized_model.parameters(), lr=1e-4)
        opt_omg  = torch.optim.Adam(self.interactive_model.parameters(),  lr=1e-4)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        # temperature + weights for KD
        T = 2.0
        alpha_logits = 0.5   # weight of KL term
        alpha_feats  = 0.5   # weight of MSE(hid) term

        train_loader = DataLoader(
            self.local_data,
            batch_size=experiment_config.batch_size,
            shuffle=True,             # shuffle please
            num_workers=0,
            drop_last=True
        )

        print(f"\nClient {self.id_} begins training\n")
        for epoch in range(experiment_config.epochs_num_train_client):
            tot_phi, tot_omg, n = 0.0, 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                n += 1

                # ====== A) Update personalized model φ (teacher = ω, frozen) ======
                with torch.no_grad():
                    logits_omg_T, feat_omg_T = self._forward_with_feat(self.interactive_model, x)
                    p_omg_T = F.softmax(logits_omg_T / T, dim=1)  # teacher probs (no grad)
                    feat_omg_T = feat_omg_T if feat_omg_T.dim() <= 2 else torch.flatten(feat_omg_T, 1)

                logits_phi, feat_phi = self._forward_with_feat(self.personalized_model, x)
                p_phi_log = F.log_softmax(logits_phi / T, dim=1)
                feat_phi = feat_phi if feat_phi.dim() <= 2 else torch.flatten(feat_phi, 1)

                L_phi_task = ce(logits_phi, y)
                KL_omg_to_phi = F.kl_div(p_phi_log, p_omg_T, reduction='batchmean') * (T * T)
                L_feat_phi = mse(feat_phi, feat_omg_T)

                L_phi = L_phi_task + alpha_logits * KL_omg_to_phi + alpha_feats * L_feat_phi
                opt_phi.zero_grad()
                L_phi.backward()
                torch.nn.utils.clip_grad_norm_(self.personalized_model.parameters(), 1.0)
                opt_phi.step()

                # ====== B) Update interactive model ω (teacher = φ, now frozen) ======
                with torch.no_grad():
                    logits_phi_T, feat_phi_T = self._forward_with_feat(self.personalized_model, x)
                    p_phi_T = F.softmax(logits_phi_T / T, dim=1)
                    feat_phi_T = feat_phi_T if feat_phi_T.dim() <= 2 else torch.flatten(feat_phi_T, 1)

                logits_omg, feat_omg = self._forward_with_feat(self.interactive_model, x)
                p_omg_log = F.log_softmax(logits_omg / T, dim=1)
                feat_omg = feat_omg if feat_omg.dim() <= 2 else torch.flatten(feat_omg, 1)

                L_omg_task = ce(logits_omg, y)
                KL_phi_to_omg = F.kl_div(p_omg_log, p_phi_T, reduction='batchmean') * (T * T)
                L_feat_omg = mse(feat_omg, feat_phi_T)

                L_omg = L_omg_task + alpha_logits * KL_phi_to_omg + alpha_feats * L_feat_omg
                opt_omg.zero_grad()
                L_omg.backward()
                torch.nn.utils.clip_grad_norm_(self.interactive_model.parameters(), 1.0)
                opt_omg.step()

                tot_phi += float(L_phi.item());
                tot_omg += float(L_omg.item())

            print(
                f"Epoch {epoch + 1}/5 | Personalized Loss: {tot_phi / max(n, 1):.4f} | Interactive Loss: {tot_omg / max(n, 1):.4f}")


        # eval (drop Top-100; it’s always 100% on 100 classes)
        self.accuracy_per_client_1[t]  = self.evaluate_accuracy_single(data_=self.local_test_set, model=self.personalized_model, k=1)
        self.accuracy_per_client_5[t]  = self.evaluate_accuracy(data_=self.local_test_set, model=self.personalized_model, k=5)
        self.accuracy_per_client_10[t] = self.evaluate_accuracy(data_=self.local_test_set, model=self.personalized_model, k=10)
        print("accuracy_per_client_1", self.accuracy_per_client_1[t])

        return self.calculate_param_variation()

    def calculate_param_variation(self):
        if self.initial_state is None:
            raise ValueError("Initial state not set. Did you call train()?")

        with torch.no_grad():
            cur = self.interactive_model.state_dict()
            # float-only, stable keys
            keys = [k for k in cur.keys() if torch.is_floating_point(cur[k])]
            return {k: (cur[k] - self.initial_state[k]).detach().clone() for k in keys}

    def update_interactive_model(self, avg_param_variations):
        # add averaged delta into the interactive model (float-only)
        with torch.no_grad():
            state = self.interactive_model.state_dict()
            for key, variation in avg_param_variations.items():
                if key not in state:
                    continue
                target = state[key]
                if not torch.is_floating_point(target):
                    continue
                if not isinstance(variation, torch.Tensor):
                    variation = torch.as_tensor(variation)
                if variation.shape != target.shape:
                    continue
                target.add_(variation.to(device=target.device, dtype=target.dtype))
            self.interactive_model.load_state_dict(state, strict=False)
        print(f"Client {self.id_}: updated interactive model with averaged parameter variations.")

class Server_pFedCK(Server):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict, clients):
        super().__init__(id_, global_data, test_data, clients_ids, clients_test_data_dict)
        self.clients = clients  # list[Client_pFedCK]

    # Make a stable, float-only flatten using sorted keys
    def _flatten_delta(self, delta_w):
        keys = sorted(delta_w.keys())
        flats = []
        for k in keys:
            v = delta_w[k]
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                flats.append(v.detach().flatten().cpu())
        if not flats:
            return torch.zeros(1)
        return torch.cat(flats)

    def calculate_cosine_similarity(self, delta_w1, delta_w2):
        f1 = self._flatten_delta(delta_w1).double().numpy()
        f2 = self._flatten_delta(delta_w2).double().numpy()
        n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(f1, f2) / (n1 * n2))

    def cluster_clients(self, delta_ws, num_clusters=5):
        # KMeans over the flattened Δω embeddings (not over the similarity matrix)
        X = []
        for dw in delta_ws:
            X.append(self._flatten_delta(dw).double().numpy())
        X = np.stack(X, axis=0)
        # guard: if fewer clients than clusters, reduce K
        k = min(num_clusters, len(delta_ws))
        if k <= 1:
            return np.zeros(len(delta_ws), dtype=int)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(X)
        return labels

    def average_parameter_variations(self, delta_ws, cluster_labels):
        cluster_avg = {}
        delta_ws = list(delta_ws)
        for cid in np.unique(cluster_labels):
            idxs = np.where(cluster_labels == cid)[0]
            # deep copy first dict, then average floats keywise
            avg = {k: v.detach().clone() for k, v in delta_ws[idxs[0]].items()}
            for k in list(avg.keys()):
                if not (isinstance(avg[k], torch.Tensor) and torch.is_floating_point(avg[k])):
                    avg.pop(k); continue
                # sum over members
                s = avg[k]
                for i in idxs[1:]:
                    s = s + delta_ws[i][k].to(device=s.device, dtype=s.dtype)
                avg[k] = s / float(len(idxs))
            cluster_avg[cid] = avg
        return cluster_avg

    def send_avg_delta_to_clients(self, cluster_avg, cluster_labels):
        for i, client in enumerate(self.clients):
            cid = int(cluster_labels[i])
            client.update_interactive_model(cluster_avg[cid])

    def cluster_and_aggregate(self, delta_ws):
        labels = self.cluster_clients(delta_ws, num_clusters=5)
        cluster_avg = self.average_parameter_variations(delta_ws, labels)
        self.send_avg_delta_to_clients(cluster_avg, labels)


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

        global_data_loader = DataLoader(self.global_data[self.current_iteration], batch_size=experiment_config.batch_size, shuffle=False)

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


    def train(self, mean_pseudo_labels,  cluster_num="0",selected_model=None):

        print(f"Mean pseudo-labels shape: {mean_pseudo_labels.shape}")  # Should be (num_data_points, num_classes)

        print(f"*** {self.__str__()} train *** Cluster: {cluster_num} ***")
        server_loader = DataLoader(self.global_data[self.current_iteration] , batch_size=experiment_config.batch_size, shuffle=False,
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
                pseudo_targets = self.apply_temperature_to_probs(pseudo_targets)

                # Check if pseudo_targets size matches the input batch size
                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}"
                    )
                    continue

                # Normalize pseudo targets to sum to 1
                #pseudo_targets = F.softmax(pseudo_targets, dim=1)

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

    def iteration_context(self,t):
        self.current_iteration = t
        mean_pseudo_labels_per_cluster_dict, self.clusters_client_id_dict_per_iter[t] = self.get_pseudo_labels_input_per_cluster(t)  # #


        for cluster_id,pseudo_labels_for_cluster in mean_pseudo_labels_per_cluster_dict.items():
            for client_id in self.clusters_client_id_dict_per_iter[t][cluster_id]:
                self.pseudo_label_to_send[client_id] = pseudo_labels_for_cluster


class Server_Centralized(Server):
    def __init__(self,id_, train_data, test_data,evaluate_every):
        LearningEntity.__init__ (self,id_,None,None)



        self.train_data =self.break_the_dict_structure(train_data)
        self.test_data  = self.break_the_dict_structure(test_data)

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




    def iteration_context(self,t):
        for cluster_id,model in self.multi_model_dict.items():
            self.fine_tune(model,cluster_id,self.train_data[cluster_id],self.test_data[cluster_id])


    def fine_tune(self,model,cluster_id,train,test):
        print("*** " + self.__str__() + " fine-tune ***")

        fine_tune_loader = DataLoader(train, batch_size=experiment_config.batch_size, shuffle=True)
        model.train()  # Set the model to training mode

        # Define loss function and optimizer

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)#experiment_config.learning_rate_train_s)

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
            if   epoch % self.evaluate_every == 0 and epoch!=0:
                self.accuracy_per_cluster_model[cluster_id][epoch] = self.evaluate_accuracy(test,model=model, k=1)

            result_to_print = epoch_loss / len(fine_tune_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {result_to_print:.4f}")
        #self.weights = self.model.state_dict()self.weights = self.model.state_dict()

    def break_the_dict_structure(self,data_):
        ans = {}
        if isinstance(experiment_config.num_clusters, int):
            if experiment_config.num_clusters !=1:
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

import copy
import torch

import copy
import torch


# ===== Vanilla FedAvg Client ===============================================
import torch
from torch import nn
from torch.utils.data import DataLoader

class ClientFedAvg(Client):
    """
    Minimal FedAvg client:
      - At t==0: start from current model (or weights_received if provided).
      - At t>0 : load weights_received (global) then train locally.
      - After training: set weights_to_send to model.state_dict().
    """
    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)
        self.weights_received = None   # filled by coordinator from server.weights_to_send
        self.weights_to_send  = None
        # Expect these dicts to exist in your base; if not, init here:
        if not hasattr(self, "size_sent"):
            self.size_sent = {}
        if not hasattr(self, "accuracy_per_client_1"):
            self.accuracy_per_client_1 = {}
        if not hasattr(self, "accuracy_per_client_5"):
            self.accuracy_per_client_5 = {}
        if not hasattr(self, "accuracy_per_client_10"):
            self.accuracy_per_client_10 = {}
        if not hasattr(self, "accuracy_per_client_100"):
            self.accuracy_per_client_100 = {}
        if not hasattr(self, "epoch_count"):
            self.epoch_count = 0
        self.current_iteration = -1

    def iteration_context(self, t: int):
        self.current_iteration = t
        # Load server global at start of round (vanilla FedAvg)
        if self.weights_received is not None:
            self.model.load_state_dict(self.weights_received)

        # Local training
        self.weights_to_send = self._local_train()

        # Track payload size (MB)
        total_bytes = 0
        for p in self.weights_to_send.values():
            if torch.is_tensor(p):
                total_bytes += p.numel() * p.element_size()
        self.size_sent[t] = total_bytes / (1024 * 1024)

        # Accuracy bookkeeping (on local test set; adapt if you prefer global)
        self.accuracy_per_client_1[t]   = self.evaluate_accuracy_single(self.local_test_set, k=1)
        self.accuracy_per_client_5[t]   = self.evaluate_accuracy(self.local_test_set, k=5)
        self.accuracy_per_client_10[t]  = self.evaluate_accuracy(self.local_test_set, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)

    # ---- local training ----
    def _local_train(self):
        print(f"*** {self} local training (FedAvg) ***")
        loader = DataLoader(self.local_data,
                            batch_size=experiment_config.batch_size,
                            shuffle=True)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=experiment_config.learning_rate_fine_tune_c)

        epochs = experiment_config.epochs_num_input_fine_tune_clients
        for epoch in range(epochs):
            self.epoch_count += 1
            running = 0.0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running += loss.item()
            print(f"[{self.id_}] epoch {epoch+1}/{epochs}  loss={running/ max(1,len(loader)):.4f}")

        return {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in self.model.state_dict().items()}

# ===== Vanilla FedAvg Server (no clustering) ===============================
import copy
import torch

class ServerFedAvg(Server):
    """
    Vanilla FedAvg server:
      - Keep one global state_dict.
      - Each round, aggregate client uploads (size-weighted).
      - Broadcast the same global model to all clients.
    Back-compat: accepts received_weights[cid] as either (state_dict, n) or plain state_dict.
    """
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict,
                 client_num_examples: dict = None):
        super().__init__(id_, global_data, test_data, clients_ids, clients_test_data_dict)
        self.received_weights = {}      # round buffer: cid -> (state_dict_on_cpu, n) OR state_dict_on_cpu
        self.weights_to_send  = {}      # cid -> global_state_dict (deepcopy)
        self.client_num_examples = client_num_examples or {cid: 1 for cid in clients_ids}
        self.current_iteration = -1
        self.global_state_dict = None   # optional snapshot

    # Optional convenience API (use in your coordinator loop if you like)
    def receive(self, client_id: str, state_dict: dict, num_examples: int = None):
        if num_examples is None:
            num_examples = int(self.client_num_examples.get(client_id, 1))
        sd_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
        self.received_weights[client_id] = (sd_cpu, int(num_examples))

    def iterate(self, t: int):
        """Aggregate one global model and prep it for broadcast to all clients."""
        self.current_iteration = t

        # Require all clients for vanilla FedAvg
        missing = set(self.clients_ids) - set(self.received_weights.keys())
        if missing:
            raise RuntimeError(f"FedAvg: missing client uploads: {sorted(missing)}")

        # Collect aligned lists
        weights_list, size_weights = [], []
        for cid in self.clients_ids:
            val = self.received_weights[cid]
            if isinstance(val, tuple) and len(val) == 2:
                sd, n = val
            else:
                # Back-compat (plain state_dict)
                sd = val
                n  = int(self.client_num_examples.get(cid, 1))

            # ensure CPU tensors
            sd_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
            weights_list.append(sd_cpu)
            size_weights.append(float(n))

        self._check_compatibility(weights_list)
        global_sd = self._average_weights(weights_list, size_weights)

        self.global_state_dict = global_sd
        self.weights_to_send = {cid: copy.deepcopy(global_sd) for cid in self.clients_ids}

        self.received_weights.clear()

    # Some frameworks call this instead
    def iteration_context(self, t: int):
        self.iterate(t)

    # ---------- helpers ----------
    def _check_compatibility(self, weights_list):
        keys0 = set(weights_list[0].keys())
        for i, w in enumerate(weights_list[1:], start=1):
            if set(w.keys()) != keys0:
                raise ValueError(f"FedAvg: state_dict keys mismatch at idx {i}")
            for k in keys0:
                a, b = weights_list[0][k], w[k]
                if torch.is_tensor(a) and torch.is_tensor(b) and a.shape != b.shape:
                    raise ValueError(f"FedAvg: tensor shape mismatch for '{k}': "
                                     f"{tuple(a.shape)} vs {tuple(b.shape)} (idx {i})")

    def _average_weights(self, weights_list, size_weights):
        total = float(sum(size_weights))
        if total <= 0:
            size_weights = [1.0] * len(size_weights)
            total = float(len(size_weights))
        norm = 1.0 / total
        scalars = [sw * norm for sw in size_weights]

        out = {}
        with torch.no_grad():
            for k in weights_list[0].keys():
                v0 = weights_list[0][k]
                if torch.is_tensor(v0) and v0.is_floating_point():
                    acc = torch.zeros_like(v0, device="cpu", dtype=torch.float32)
                    for w, s in zip(weights_list, scalars):
                        acc.add_(w[k].to(dtype=torch.float32), alpha=s)
                    out[k] = acc.to(dtype=v0.dtype)
                else:
                    # ints/bools (e.g., num_batches_tracked) → copy from first
                    out[k] = copy.deepcopy(v0)
        return out

    def get_weights_for(self, client_id: str):
        return self.weights_to_send.get(client_id, None)

# ---------------- FedBABU: Client & Server ----------------
# Assumes your imports / globals are already present:
# import torch, torch.nn as nn, torch.nn.functional as F
# from torch.utils.data import DataLoader
# device, experiment_config, DataSet, etc. exist
# Base classes: Client, Server (or ServerFedAvg)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FedBABU drop-in implementation:
- Robust head detection (Linear or 1x1 Conv head) via param shapes (out_dim == num_classes)
- Client: re-init + freeze head, train body-only → upload body; optional head-only personalization
- Server: size-weighted average of BODY-only state_dicts

Usage (matches your main loop pattern):
    ensure_fedbabu_params(experiment_config)

    clients, clients_ids, clients_test_by_id_dict = create_clients(
        clients_train_data_dict, server_train_data, clients_test_data_dict, server_test_data
    )

    server = Server_FedBABU(
        id_="server",
        global_data=server_train_data,
        test_data=server_test_data,
        clients_ids=clients_ids,
        clients_test_data_dict=clients_test_by_id_dict,
        clients=clients,
    )

    g0 = copy.deepcopy(server.global_state)  # BODY-only dict
    for c in clients:
        c.set_model_from_global(g0)

    for t in range(experiment_config.iterations):
        print("---------------------------- iter:", t)
        server.run_round(t)
        rd = RecordData(clients, server)
        save_record_to_results(rd)
"""

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Expect these to be defined by your codebase:
# - class Client, class Server
# - global `device` (torch.device)
# - global `experiment_config` (namespace/obj with your usual fields)


# ---------------------------------------------------------------------
# Ensure FedBABU hyperparameters exist on experiment_config
# (Defaults derive from your existing names to keep behavior familiar)
# ---------------------------------------------------------------------
def ensure_fedbabu_params(ec):
    # Required base params you already use
    assert hasattr(ec, "batch_size")
    assert hasattr(ec, "learning_rate_train_c")
    assert hasattr(ec, "learning_rate_fine_tune_c")
    assert hasattr(ec, "epochs_num_train_client")
    assert hasattr(ec, "epochs_num_input_fine_tune_clients")

    # New knobs (only add if missing)
    if not hasattr(ec, "learning_rate_fedbabu_body"):
        ec.learning_rate_fedbabu_body = float(ec.learning_rate_train_c)
    if not hasattr(ec, "epochs_num_fedbabu_body"):
        ec.epochs_num_fedbabu_body = int(ec.epochs_num_train_client)

    if not hasattr(ec, "learning_rate_fedbabu_head"):
        ec.learning_rate_fedbabu_head = float(ec.learning_rate_fine_tune_c)
    if not hasattr(ec, "epochs_num_fedbabu_head"):
        ec.epochs_num_fedbabu_head = max(1, int(ec.epochs_num_input_fine_tune_clients // 2))

    if not hasattr(ec, "fedbabu_personalize_each_round"):
        ec.fedbabu_personalize_each_round = True

    if not hasattr(ec, "fedbabu_grad_clip_norm"):
        ec.fedbabu_grad_clip_norm = 1.0

    # Helpful but optional; if models expose num_classes, we can skip.
    # If you know your dataset, set ec.num_classes elsewhere.
    # if not hasattr(ec, "num_classes"): pass


# ------------------------------ Client (FedBABU) ------------------------------ #
class Client_FedBABU(Client):
    """
    FedBABU Client:
      Round t:
        1) set_model_from_global(global_body): load BODY weights, keep (and then re-init) HEAD
        2) re-init HEAD, freeze HEAD, unfreeze BODY, train BODY-only on local CE
        3) upload BODY-only state_dict to server
        4) (optional) personalize: freeze BODY, unfreeze HEAD, train HEAD-only for eval
    """

    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)
        self._fedbabu_init_done = False
        self._head_prefix: str | None = None     # e.g., "classifier.6"
        self._body_keys: set[str] = set()
        self.w_model: nn.Module | None = None
        self._init_head_body_partitions()

    # ---- detection of head/body ------------------------------------------------
    def _infer_num_classes(self):
        ec = experiment_config
        if hasattr(ec, "num_classes") and ec.num_classes is not None:
            return int(ec.num_classes)
        # fallback to model attribute if present
        if hasattr(self.model, "num_classes") and self.model.num_classes is not None:
            return int(self.model.num_classes)
        raise RuntimeError("[FedBABU] could not infer num_classes (set experiment_config.num_classes).")

    def _init_head_body_partitions(self):
        """
        Identify the classifier 'head' via parameter shapes:
          - Linear: weight.shape[0] == num_classes
          - Conv2d: weight.shape[0] == num_classes (often kh=kw=1)
        Falls back to last Linear module name, else last param prefix.
        """
        num_classes = self._infer_num_classes()
        sd = self.model.state_dict()
        param_keys = [k for k in sd.keys() if k.endswith(".weight") or k.endswith(".bias")]

        candidates = []
        for k in param_keys:
            wkey = k if k.endswith(".weight") else k[:-5] + ".weight"
            if wkey in sd:
                w = sd[wkey]
                if w.ndim == 2 and w.shape[0] == num_classes:          # Linear
                    candidates.append(wkey[:-len(".weight")])
                elif w.ndim == 4 and w.shape[0] == num_classes:        # Conv2d
                    candidates.append(wkey[:-len(".weight")])

        if candidates:
            head_prefix = candidates[-1]
        else:
            # fallback 1: last Linear module
            last_linear = None
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Linear):
                    last_linear = name
            if last_linear is not None:
                head_prefix = last_linear
            else:
                # fallback 2: last param prefix
                if not param_keys:
                    raise RuntimeError("[FedBABU] no parameters found; cannot choose head.")
                head_prefix = param_keys[-1].rsplit(".", 1)[0]

        # Cache partition
        head_keys = set()
        for k in sd.keys():
            if k == head_prefix or k.startswith(head_prefix + "."):
                head_keys.add(k)
        body_keys = set(sd.keys()) - head_keys
        if not body_keys:
            raise RuntimeError("[FedBABU] body detected as empty; head detection failed.")

        self._head_prefix = head_prefix
        self._body_keys = body_keys
        self._fedbabu_init_done = True
        print(f"[FedBABU][client {self.id_}] head_prefix={self._head_prefix} | body_keys={len(self._body_keys)}")

    # ---- model (de)serialization helpers --------------------------------------
    def _get_body_state(self, model) :
        mdl = self.model if model is None else model
        full = mdl.state_dict()
        return {k: v.detach().clone() for k, v in full.items() if k in self._body_keys}

    def _load_body_state(self, body_sd, model):
        mdl = self.model if model is None else model
        with torch.no_grad():
            cur = mdl.state_dict()
            for k, v in body_sd.items():
                if k in cur:
                    if torch.is_tensor(cur[k]):
                        cur[k].copy_(v.to(dtype=cur[k].dtype, device=cur[k].device))
                    else:
                        cur[k] = copy.deepcopy(v)
            mdl.load_state_dict(cur, strict=False)

    # ---- required by your main loop -------------------------------------------
    def set_model_from_global(self, global_state):
        """
        Called by the coordinator BEFORE each round.
        In FedBABU, global_state is BODY-only. We copy it into our local model (BODY only).
        """
        if not self._fedbabu_init_done:
            self._init_head_body_partitions()

        self.w_model = self.model
        if global_state is not None:
            self._load_body_state(global_state, model=self.w_model)

    # ---- FedBABU local steps ---------------------------------------------------
    def _reinit_head(self):
        """
        Re-initialize the classifier head (supports Linear or Conv2d).
        """
        if not self._head_prefix:
            return
        module = dict(self.w_model.named_modules()).get(self._head_prefix, None)

        if module is None:
            # Reinit by touching state_dict directly
            sd = self.w_model.state_dict()
            wkey = self._head_prefix + ".weight"
            bkey = self._head_prefix + ".bias"
            with torch.no_grad():
                if wkey in sd:
                    if sd[wkey].ndim == 2:
                        nn.init.normal_(sd[wkey], mean=0.0, std=0.02)  # Linear
                    elif sd[wkey].ndim == 4:
                        nn.init.kaiming_normal_(sd[wkey])              # Conv2d
                if bkey in sd:
                    sd[bkey].zero_()
            self.w_model.load_state_dict(sd, strict=False)
            return

        with torch.no_grad():
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _freeze_head_unfreeze_body(self):
        hp = self._head_prefix
        for name, p in self.w_model.named_parameters():
            if hp and (name == hp or name.startswith(hp + ".")):
                p.requires_grad = False
            else:
                p.requires_grad = True

    def _freeze_body_unfreeze_head(self):
        hp = self._head_prefix
        for name, p in self.w_model.named_parameters():
            if hp and (name == hp or name.startswith(hp + ".")):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _train_body_only(self):
        """
        CE on local data; ONLY body params have requires_grad=True.
        """
        self.w_model.train()
        loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size,
                            shuffle=True, num_workers=0, drop_last=False)
        criterion = nn.CrossEntropyLoss()
        body_params = [p for p in self.w_model.parameters() if p.requires_grad]
        lr = getattr(experiment_config, "learning_rate_fedbabu_body",
                     getattr(experiment_config, "learning_rate_fine_tune_c", 1e-3))
        optimizer = torch.optim.Adam(body_params, lr=lr)

        epochs = getattr(experiment_config, "epochs_num_fedbabu_body",
                         getattr(experiment_config, "epochs_num_train_client", 1))
        clip = getattr(experiment_config, "fedbabu_grad_clip_norm", 1.0)

        last = float("nan")
        for ep in range(epochs):
            ep_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.w_model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(body_params, clip)
                optimizer.step()
                ep_loss += float(loss.item())
            last = ep_loss / max(1, len(loader))
            print(f"[Client {self.id_}][FedBABU][body] epoch {ep+1}/{epochs} loss={last:.4f}")
        return last

    def _personalize_head_only(self):
        """
        Optional: fine-tune head only (for evaluation).
        """
        self.w_model.train()
        loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size,
                            shuffle=True, num_workers=0, drop_last=False)
        criterion = nn.CrossEntropyLoss()
        self._freeze_body_unfreeze_head()
        head_params = [p for p in self.w_model.parameters() if p.requires_grad]

        lr = getattr(experiment_config, "learning_rate_fedbabu_head",
                     getattr(experiment_config, "learning_rate_fine_tune_c", 1e-3))
        optimizer = torch.optim.Adam(head_params, lr=lr)
        epochs = getattr(experiment_config, "epochs_num_fedbabu_head",
                         max(1, getattr(experiment_config, "epochs_num_input_fine_tune_clients", 2) // 2))
        clip = getattr(experiment_config, "fedbabu_grad_clip_norm", 1.0)

        last = float("nan")
        for ep in range(epochs):
            ep_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.w_model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, clip)
                optimizer.step()
                ep_loss += float(loss.item())
            last = ep_loss / max(1, len(loader))
            print(f"[Client {self.id_}][FedBABU][head] epoch {ep+1}/{epochs} loss={last:.4f}")
        return last

    # ---- entry point called by server.run_round --------------------------------
    def train(self, t):
        """
        Run local FedBABU steps for round t.
        Returns:
            body_state_dict (BODY-only), num_examples (int)
        """
        # Step A: reinit head, freeze head, train body only
        self._reinit_head()
        self._freeze_head_unfreeze_body()
        _ = self._train_body_only()

        # Prepare upload (BODY-only)
        body_sd = self._get_body_state(self.w_model)
        num_examples = len(self.local_data)

        # Optional personalization for evaluation
        if getattr(experiment_config, "fedbabu_personalize_each_round", True):
            _ = self._personalize_head_only()

        # Optional eval logging (guard if helpers absent)
        try:
            self.accuracy_per_client_1[t] = self.evaluate_accuracy_single(self.local_test_set, k=1)
            self.accuracy_per_client_5[t] = self.evaluate_accuracy(self.local_test_set, k=5)
            self.accuracy_per_client_10[t] = self.evaluate_accuracy(self.local_test_set, k=10)
            self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, k=100)
        except Exception:
            pass

        return body_sd, int(num_examples)

    def __str__(self):
        return f"Client_FedBABU {self.id_}"


# ------------------------------ Server (FedBABU) ------------------------------ #
class Server_FedBABU(Server):
    """
    FedBABU Server:
      - Maintains a single global BODY state_dict: self.global_state
      - Per round:
          broadcast BODY (self.global_state) to clients
          collect BODY updates + sizes
          size-weighted average → new BODY (self.global_state)
    """

    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict,
                 clients: List[Client_FedBABU]):
        super().__init__(id_, global_data, test_data, clients_ids, clients_test_data_dict)
        if not clients:
            raise ValueError("Server_FedBABU needs a non-empty client list")
        self.clients: List[Client_FedBABU] = clients
        # Initialize global BODY from a probe client
        probe_client = clients[0]
        self.global_state: Dict[str, torch.Tensor] = probe_client._get_body_state(probe_client.model)
        self.current_iteration = -1

    @property
    def global_state_dict(self):
        """Alias if other parts of your code expect this name."""
        return self.global_state

    def _aggregate_weighted(self, bodies,
                            sizes) :
        total = float(sum(sizes))
        if total <= 0:
            sizes = [1.0] * len(sizes)
            total = float(len(sizes))
        scalars = [s / total for s in sizes]

        keys = list(bodies[0].keys())
        out = {}
        with torch.no_grad():
            for k in keys:
                v0 = bodies[0][k]
                if torch.is_tensor(v0) and v0.is_floating_point():
                    acc = torch.zeros_like(v0, device="cpu", dtype=torch.float32)
                    for sd, a in zip(bodies, scalars):
                        acc.add_(sd[k].to(dtype=torch.float32, device="cpu"), alpha=a)
                    out[k] = acc.to(dtype=v0.dtype)
                else:
                    out[k] = copy.deepcopy(v0)
        return out

    def run_round(self, t):
        """
        One FL round:
          - broadcast BODY (self.global_state) to clients
          - client executes local FedBABU steps and returns BODY + num_examples
          - aggregate to update self.global_state
        """
        self.current_iteration = t

        # Broadcast BODY to clients
        for c in self.clients:
            c.set_model_from_global(self.global_state)

        # Collect uploads (move to CPU for aggregation)
        bodies, sizes = [], []
        for c in self.clients:
            body_sd, n = c.train(t)
            body_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in body_sd.items()}
            bodies.append(body_cpu)
            sizes.append(float(n))

        # Validate key sets
        ref_keys = set(bodies[0].keys())
        for i, sd in enumerate(bodies[1:], start=1):
            if set(sd.keys()) != ref_keys:
                raise ValueError(f"[FedBABU][server] BODY keys mismatch at client index {i}")

        # Aggregate → new BODY
        self.global_state = self._aggregate_weighted(bodies, sizes)

    # Optional: global eval (not canonical; global head isn’t meaningful in FedBABU)
    def eval_global_top1(self) :
        try:
            probe = self.clients[0]
            temp = probe.get_client_model() if hasattr(probe, "get_client_model") else copy.deepcopy(probe.model)
            with torch.no_grad():
                cur = temp.state_dict()
                for k, v in self.global_state.items():
                    if k in cur and torch.is_tensor(cur[k]):
                        cur[k].copy_(v.to(dtype=cur[k].dtype, device=cur[k].device))
                    elif k in cur:
                        cur[k] = copy.deepcopy(v)
                temp.load_state_dict(cur, strict=False)

            loader = DataLoader(self.test_global_data, batch_size=128, shuffle=False, num_workers=0)
            temp.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits = temp(x)
                    pred = logits.argmax(dim=1)
                    correct += int((pred == y).sum().item())
                    total += int(y.numel())
            return 100.0 * correct / max(1, total)
        except Exception as e:
            print(f"[FedBABU][server] eval_global_top1 failed: {e}")
            return float("nan")

    def __str__(self):
        return "Server_FedBABU"



# ===== Helpers =====
def _named_param_dict(model):
    """Return a dict: name -> learnable Parameter (no buffers)."""
    return {name: p for name, p in model.named_parameters()}

def _set_bn_eval(m):
    """Freeze BatchNorm running stats (keeps affine params trainable)."""
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



@torch.no_grad()
def _quick_eval_top1(model, dataset, batch_size=64, max_batches=999999):
    """Evaluation that *only* calls model(x) — no cluster_id or framework tricks."""
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    correct = total = 0
    for i, (x, y) in enumerate(dl):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.numel()
    return (100.0 * correct / max(total, 1))

# ===== pFedMe Client =====
# ===== pFedMe Client =====
# ================= pFedMe Client =================
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ===== helpers (keep these once in your file) =====
def _named_param_dict(model):
    return {name: p for name, p in model.named_parameters()}

def _set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

@torch.no_grad()
def _quick_eval_top1(model, dataset, batch_size=64, max_batches=999999):
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    correct = total = 0
    for i, (x, y) in enumerate(dl):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.numel()
    return 100.0 * correct / max(total, 1)

# ===== pFedMe client with batch-resampled inner loop =====
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ===== helpers (keep these once in your file) =====
def _named_param_dict(model):
    return {name: p for name, p in model.named_parameters()}

def _set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

@torch.no_grad()
def _quick_eval_top1(model, dataset, batch_size=64, max_batches=999999):
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    correct = total = 0
    for i, (x, y) in enumerate(dl):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.numel()
    return 100.0 * correct / max(total, 1)

# ===== pFedMe client with batch-resampled inner loop =====
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# expects helpers defined once in your file:
#   _named_param_dict(model), _set_bn_eval(m), _quick_eval_top1(model, dataset, ...)

class Client_pFedMe(Client):
    """
    pFedMe client:
      Inner loop:    θ ← argmin CE(θ; D_batch) + (λ/2)||θ - w||^2   (K steps, fresh batch each step)
      Outer update:  w ← w - μ (w - θ)

    Notes
    - Uses UNNORMALIZED L2 prox (as in pFedMe).
    - Freezes BN running stats during θ steps (affine params still trainable).
    - Clips grads at 5.0; prints raw (pre-clip) grad norm for first few steps on client 0, round 0.
    - Honors your experiment_config; adds safe floors + an explicit μ knob for easier tuning.
    """

    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)

        # Local copy of the global model
        self.w_model,ttt = self.get_client_model()

        # ---- Hyperparams (reuse your config; add guardrails/defaults) ----
        # minibatch (allow client-specific override)
        base_mb           = getattr(experiment_config, "batch_size", 32)
        self.mb           = int(getattr(experiment_config, "pfedme_batch_size", base_mb))

        # outer steps per round (reuse your "epochs_num_train_client")
        self.R            = int(getattr(experiment_config, "epochs_num_train_client", 5))

        # inner θ steps per outer step (ensure a minimum bite)
        self.K            = max(5, int(getattr(experiment_config, "inner_steps", 1)))

        # inner θ learning rate
        base_theta_lr     = getattr(experiment_config, "pfedme_theta_lr",
                                    getattr(experiment_config, "learning_rate_train_c", 1e-3))
        self.theta_lr     = max(1e-3, float(base_theta_lr))

        # proximal strength λ
        self.lam          = float(getattr(experiment_config, "pfedme_lambda",
                                   getattr(experiment_config, "lambda_consistency", 5.0)))

        # outer step size μ (direct knob). Default: eta*lambda or learning_rate_train_c*lambda
        default_eta       = getattr(experiment_config, "pfedme_eta",
                                    getattr(experiment_config, "learning_rate_train_c", 1e-3))
        self.mu           = float(getattr(experiment_config, "pfedme_mu", default_eta * self.lam))

        self.criterion_ce = nn.CrossEntropyLoss()

    # ----- API used by the server -----
    def set_model_from_global(self, global_state):
        """Server broadcasts global state; copy into local w."""
        self.w_model.load_state_dict(global_state)

    @torch.no_grad()
    def _w_update_from_theta(self, theta_model):
        """Outer update: w <- w - μ (w - θ̃) (learnable params only)."""
        w_params  = _named_param_dict(self.w_model)
        th_params = _named_param_dict(theta_model)
        for k, w in w_params.items():
            if k in th_params:
                w.add_( - self.mu * (w - th_params[k].to(w.device, w.dtype)) )

    def _assert_shapes_once(self):
        """One-shot structural checks to catch head/label issues early."""
        dl = DataLoader(self.local_data, batch_size=min(8, self.mb), shuffle=True, num_workers=0, drop_last=False)
        x, y = next(iter(dl))
        x, y = x.to(device), y.to(device)
        self.w_model.eval()
        with torch.no_grad():
            logits = self.w_model(x)
        assert logits.ndim == 2, f"logits ndim={logits.ndim}"
        C = logits.size(1)
        ymin, ymax = int(y.min().item()), int(y.max().item())
        assert ymin >= 0 and ymax < C, f"labels out of range [{ymin},{ymax}] vs C={C}"
        if not torch.isfinite(logits).all():
            raise RuntimeError("logits NaN/Inf")
        print(f"[shape-ok] batch={tuple(x.shape)}  num_classes={C}  label_range=[{ymin},{ymax}]")

    def _theta_inner_minimize(self, w_snapshot, data_iter, do_debug=False):
        """
        θ̃ ≈ argmin_θ  CE(θ; fresh minibatches) + (λ/2)||θ - w||^2 over K steps.
        - BN running stats frozen (affine trainable).
        - UNNORMALIZED prox (matches pFedMe).
        - Grad clip at 5.0; print raw pre-clip grad norm if do_debug.
        """
        # θ starts from w
        theta,ttt = self.get_client_model()
        theta.load_state_dict(w_snapshot)
        theta.apply(_set_bn_eval)   # freeze BN running stats
        theta.train()

        opt_theta = torch.optim.Adam(theta.parameters(), lr=self.theta_lr)

        # reference w params (same structure) for prox term
        w_ref,ttt = self.get_client_model()
        w_ref.load_state_dict(w_snapshot)
        w_params = _named_param_dict(w_ref)

        for step in range(self.K):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(DataLoader(self.local_data, batch_size=self.mb, shuffle=True,
                                            num_workers=0, drop_last=False))
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            opt_theta.zero_grad(set_to_none=True)
            logits = theta(x)
            loss_ce = self.criterion_ce(logits, y)

            # UNNORMALIZED L2 prox over learnable params
            prox = torch.zeros((), device=device)
            th_params = _named_param_dict(theta)
            for k, th_v in th_params.items():
                if k not in w_params:
                    continue
                diff = th_v - w_params[k].to(th_v.device, th_v.dtype)
                prox  = prox + diff.pow(2).sum()

            loss = loss_ce + 0.5 * self.lam * prox
            if torch.isnan(loss) or torch.isinf(loss):
                if do_debug:
                    print("[warn] θ loss NaN/Inf — skip step")
                continue

            loss.backward()

            # raw (pre-clip) grad norm for diagnostics
            raw_norm = None
            if do_debug:
                g2_raw = 0.0
                for p in theta.parameters():
                    if p.grad is not None:
                        g2_raw += float(p.grad.pow(2).sum().item())
                raw_norm = g2_raw ** 0.5

            torch.nn.utils.clip_grad_norm_(theta.parameters(), 5.0)
            opt_theta.step()

            if do_debug:
                print(f"[θ-step dbg] CE={loss_ce.item():.4f} prox={float(prox):.6f} raw_grad_norm={raw_norm:.4f}")

        return theta

    # ----- One FL round on the client -----
    def train(self, t):
        self.current_iteration = t
        self.w_model.train()

        # One-time structural checks
        if t == 0 and self.id_ in (0, "0"):
            self._assert_shapes_once()

        last_theta = None
        for r in range(self.R):
            # fresh shuffled iterator EACH outer step
            train_loader = DataLoader(self.local_data, batch_size=self.mb, shuffle=True,
                                      num_workers=0, drop_last=False)
            data_iter = iter(train_loader)

            # snapshot current w
            w_snapshot = copy.deepcopy(self.w_model.state_dict())

            # debug first few inner steps for client 0, round 0
            do_debug = (t == 0 and self.id_ in (0, "0") and r < 3)
            theta = self._theta_inner_minimize(w_snapshot, data_iter, do_debug=do_debug)
            last_theta = theta

            # outer update w <- w - μ (w - θ̃)
            self._w_update_from_theta(theta)

        # robust eval of θ (no framework wrappers)
        top1_quick = _quick_eval_top1(last_theta, self.local_test_set, batch_size=64)
        print(f"[quick-eval] client {self.id_} top-1: {top1_quick:.2f}%")

        # keep your framework metrics too
        last_theta.eval()
        self.accuracy_per_client_1[t]   = self.evaluate_accuracy_single(self.local_test_set, model=last_theta, k=1)
        self.accuracy_per_client_5[t]   = self.evaluate_accuracy(self.local_test_set, model=last_theta, k=5)
        self.accuracy_per_client_10[t]  = self.evaluate_accuracy(self.local_test_set, model=last_theta, k=10)
        self.accuracy_per_client_100[t] = self.evaluate_accuracy(self.local_test_set, model=last_theta, k=100)
        print(f"accuracy_per_client_1 {self.accuracy_per_client_1[t]:.4f}")

        # payload to server (local w after R outer steps)
        local_state = copy.deepcopy(self.w_model.state_dict())

        # log payload size (MB)
        total_size = sum(p.numel() * p.element_size() for p in self.w_model.parameters())
        self.size_sent[t] = total_size / (1024 * 1024)

        return local_state


# ===== pFedMe Server (model-free; holds a global_state dict) =====
# ===== pFedMe Server (model-free; holds a global_state dict) =====
# =================
# pFedMe Server (model-free; holds a global_state dict) =================
class Server_pFedMe(Server):
    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict, clients):
        super().__init__(id_, global_data, test_data, clients_ids, clients_test_data_dict)
        self.clients = clients

        # EMA coefficient for server update; 1.0 == simple FedAvg of local states
        self.beta = float(getattr(experiment_config, "pfedme_beta", 1.0))

        # Initialize global params from the first client's architecture
        self.global_state = copy.deepcopy(clients[0].w_model.state_dict())

        # Optional logs
        self.accuracy_server_test_1_global = {}
        self.received_states = {}

        # Optional: weighted averaging by data size (dict id->count), else simple mean
        self.client_num_examples = getattr(self, "client_num_examples", None)

    def _mean_state(self, states, weights=None):
        """(Weighted) mean over float tensors; copy non-floats from the first state."""
        keys = states[0].keys()
        out = {}
        if weights is None:
            for k in keys:
                v0 = states[0][k]
                if isinstance(v0, torch.Tensor) and v0.is_floating_point():
                    stack = [st[k].to(v0.device, v0.dtype) for st in states]
                    out[k] = torch.stack(stack, dim=0).mean(dim=0)
                else:
                    out[k] = v0
            return out

        w = torch.as_tensor(weights, dtype=torch.float32)
        w = (w / (w.sum() + 1e-12)).tolist()
        for k in keys:
            v0 = states[0][k]
            if isinstance(v0, torch.Tensor) and v0.is_floating_point():
                acc = torch.zeros_like(v0)
                for st, wi in zip(states, w):
                    acc.add_(wi * st[k].to(v0.device, v0.dtype))
                out[k] = acc
            else:
                out[k] = v0
        return out

    def run_round(self, t, selected_clients=None):
        if selected_clients is None:
            selected_clients = self.clients

        # ---- Broadcast ----
        gstate = copy.deepcopy(self.global_state)

        # ---- Local pFedMe solves ----
        local_states, weights = [], None
        for c in selected_clients:
            c.set_model_from_global(gstate)
            st = c.train(t)     # returns w_i after R outer steps
            local_states.append(st)
            self.received_states[c.id_] = st

        if self.client_num_examples:
            weights = [self.client_num_examples.get(c.id_, 1.0) for c in selected_clients]

        # ---- Aggregate local states ----
        mean_state = self._mean_state(local_states, weights)

        # ---- Global EMA update: w_{t+1} = (1-β) w_t + β * mean_i w_{i,R} ----
        for k, v in self.global_state.items():
            mv = mean_state.get(k)
            if mv is None:
                continue
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                self.global_state[k] = (1.0 - self.beta) * v + self.beta * mv.to(v.device, v.dtype)
            else:
                self.global_state[k] = v

        # (Optional) global evaluation by instantiating a temp client model:
        # tmp = selected_clients[0].get_client_model().to(device)
        # tmp.load_state_dict(self.global_state, strict=True)
        # self.accuracy_server_test_1_global[t] = self.evaluate_accuracy_single(self.test_global_data, model=tmp, k=1)

class ServerFedCT(Server):
    """
    FedCT-style server:
    - Receives hard labels (int tensor of shape [|U|]) from each client.
    - Forms a consensus labeling via majority vote (optionally with a threshold).
    - Broadcasts the consensus hard labels back to all clients as pseudo-labels.
    - No server model is trained.

    Logging:
      - self.accuracy_global_data_1[t] : consensus accuracy on global_data
      - self.accuracy_server_test_1[t] : consensus accuracy on test_global_data
    """

    def __init__(self, id_, global_data, test_data, clients_ids, clients_test_data_dict):
        super().__init__(id_, global_data, test_data, clients_ids, clients_test_data_dict)

        # Override / simplify for FedCT
        self.consensus_history = {}   # t -> tensor of shape [|U|]
        self.consensus_accuracy = {}  # t -> scalar (%), optional metric

        # FedCT does not use a server model
        self.model = None
        if hasattr(self, "multi_model_dict"):
            self.multi_model_dict = {}

        # For FedCT we want clean, flat logging dicts keyed by t
        self.accuracy_global_data_1 = {}
        self.accuracy_server_test_1 = {}

    # --- communication API ---

    def receive_single_pseudo_label(self, sender, info):
        """
        For FedCT, `info` is a 1D tensor of hard labels: shape [num_data_points].
        """
        self.pseudo_label_received[sender] = info

    # --- majority vote ---

    def majority_vote(self, votes_2d):
        """
        Majority vote over hard labels.

        Args:
            votes_2d: torch.LongTensor of shape [num_clients, num_points]

        Returns:
            consensus: torch.LongTensor of shape [num_points]
        """
        num_clients, num_points = votes_2d.shape
        num_classes = experiment_config.num_classes
        threshold = getattr(experiment_config, "fedct_majority_threshold", 0.5)

        device_local = votes_2d.device
        consensus = torch.empty(num_points, dtype=torch.long, device=device_local)

        for i in range(num_points):
            labels_i = votes_2d[:, i]
            counts = torch.bincount(labels_i, minlength=num_classes)
            max_count, max_label = torch.max(counts, dim=0)

            if max_count.float() / num_clients >= threshold:
                consensus[i] = max_label
            else:
                # Still choose majority label (you could set -1 here to "abstain")
                consensus[i] = max_label

        return consensus

    # --- logging helpers ---

    def _compute_accuracy_from_vector(self, preds_vec, dataset):
        """
        Compare an integer prediction vector with labels from a dataset.
        Returns accuracy in percent.
        """
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        _, y = next(iter(loader))
        y = y.to(preds_vec.device)

        preds = preds_vec[:y.size(0)]
        acc = (preds == y).float().mean().item() * 100.0
        return acc

    def _log_consensus_metrics(self, consensus, t):
        """
        Log:
          - accuracy_global_data_1[t]
          - accuracy_server_test_1[t]
        And store consensus_accuracy[t] for convenience.
        """
        # Accuracy on global public data U
        try:
            acc_global = self._compute_accuracy_from_vector(consensus, self.global_data)
        except Exception as e:
            print(f"[FedCT] Failed to evaluate global_data accuracy at round {t}: {e}")
            acc_global = float("nan")

        # Accuracy on global test data
        try:
            acc_test = self._compute_accuracy_from_vector(consensus, self.test_global_data)
        except Exception as e:
            print(f"[FedCT] Failed to evaluate test_global_data accuracy at round {t}: {e}")
            acc_test = float("nan")

        # Flat, t-indexed logs
        self.accuracy_global_data_1[t] = acc_global
        self.accuracy_server_test_1[t] = acc_test

        # Extra helper
        self.consensus_accuracy[t] = acc_global

        print(f"[FedCT][Server] Round {t}: "
              f"consensus acc global_data={acc_global:.2f}%, "
              f"test_global_data={acc_test:.2f}%")

    # --- main FedCT iteration ---

    def iteration_context(self, t):
        """
        One FedCT communication round:
        - Stack all clients' hard labels on U.
        - Majority vote to form consensus.
        - Log consensus accuracy.
        - Broadcast consensus back to all clients as pseudo-labels.
        """
        self.current_iteration = t
        self.pseudo_label_to_send = {}

        # Stack clients' hard labels
        client_ids = list(self.clients_ids)
        label_tensors = []
        for cid in client_ids:
            labels = self.pseudo_label_received[cid]
            if labels is None:
                raise ValueError(f"[FedCT] Missing hard labels from client {cid} at round {t}.")
            label_tensors.append(labels.to(device))

        votes = torch.stack(label_tensors, dim=0)  # [num_clients, num_points]

        # Majority-vote consensus
        consensus = self.majority_vote(votes)
        self.consensus_history[t] = consensus.detach().cpu()

        # Log metrics
        self._log_consensus_metrics(consensus, t)

        # Broadcast consensus hard labels to all clients
        for cid in client_ids:
            self.pseudo_label_to_send[cid] = consensus  # (LongTensor, [|U|])

        # Prepare for next round
        self.reset_clients_received_pl()

class ClientFedCT(Client):
    """
    FedCT-style client:
    - Same local model as your usual Client.
    - At each round:
        * If t > 0: train on pseudo-labeled global data U (hard labels from server).
        * Fine-tune on local labeled data.
        * Predict hard labels on U and send them to the server.
    Logging is aligned with base Client:
        - accuracy_per_client_1[t], _5[t], _10[t], _100[t]
    """

    def __init__(self, id_, client_data, global_data, global_test_data, local_test_data):
        super().__init__(id_, client_data, global_data, global_test_data, local_test_data)

    # --- FedCT helpers ---

    def predict_hard_labels_on_global(self):
        self.model.eval()
        preds = []

        loader = DataLoader(self.global_data,
                            batch_size=experiment_config.batch_size,
                            shuffle=False,
                            num_workers=0)

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                hard = outputs.argmax(dim=1)
                preds.append(hard.cpu())

        preds = torch.cat(preds, dim=0).long()
        print(f"[FedCT][Client {self.id_}] hard labels shape on global: {preds.shape}")
        return preds

    def train_on_pseudo_labeled_global(self, hard_labels):
        print(f"[FedCT][Client {self.id_}] Train on pseudo-labeled global data. "
              f"Labels shape: {tuple(hard_labels.shape)}")

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=experiment_config.learning_rate_train_c)

        hard_labels = hard_labels.to(device)

        loader = DataLoader(self.global_data,
                            batch_size=experiment_config.batch_size,
                            shuffle=False,
                            num_workers=0,
                            drop_last=True)

        num_epochs = experiment_config.epochs_num_train_client
        last_avg_loss = 0.0

        for epoch in range(num_epochs):
            self.epoch_count += 1
            epoch_loss = 0.0
            processed = 0

            for batch_idx, (inputs, _) in enumerate(loader):
                inputs = inputs.to(device)
                bs = inputs.size(0)

                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + bs
                if end_idx > hard_labels.size(0):
                    print(f"[FedCT][Client {self.id_}] Skipping batch {batch_idx} "
                          f"due to label length mismatch.")
                    continue

                targets = hard_labels[start_idx:end_idx]

                optimizer.zero_grad()
                outputs = self.model(inputs)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)

                loss = criterion(outputs, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[FedCT][Client {self.id_}] NaN/Inf loss at batch {batch_idx}. Skipping.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                processed += 1

            if processed > 0:
                last_avg_loss = epoch_loss / processed
            else:
                last_avg_loss = float("nan")

            print(f"[FedCT][Client {self.id_}] Epoch [{epoch+1}/{num_epochs}], "
                  f"Loss: {last_avg_loss:.4f}")

        return last_avg_loss

    # --- main FedCT iteration ---

    def iteration_context(self, t):
        """
        FedCT round t:
        - If t > 0 and pseudo-labels from server are available, train on U with hard labels.
        - Fine-tune on local data.
        - Send new hard labels on U to server.
        - Log local accuracies in the same structures as base Client.
        """
        self.current_iteration = t
        print(f"[FedCT][Client {self.id_}] --- Iteration {t} ---")

        # 1) Train on pseudo-labeled global data
        if t > 0 and self.pseudo_label_received is not None:
            self.train_on_pseudo_labeled_global(self.pseudo_label_received)

        # 2) Fine-tune on local labeled data
        self.fine_tune()

        # 3) Send new hard labels
        hard_preds = self.predict_hard_labels_on_global()
        self.pseudo_label_to_send = hard_preds

        # 4) Log accuracies (same dict names as in your base Client)
        self.accuracy_per_client_1[self.current_iteration] = \
            self.evaluate_accuracy_single(self.local_test_set, k=1)

        self.accuracy_per_client_5[self.current_iteration] = \
            self.evaluate_accuracy(self.local_test_set, k=5)

        self.accuracy_per_client_10[self.current_iteration] = \
            self.evaluate_accuracy(self.local_test_set, k=10)

        self.accuracy_per_client_100[self.current_iteration] = \
            self.evaluate_accuracy(self.local_test_set, k=100)

        print(f"[FedCT][Client {self.id_}] Done iteration {t}")
