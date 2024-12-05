import threading
import random

import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
import torch.nn.functional as F

from abc import ABC, abstractmethod

# Define AlexNet for clients

def get_file_name(server_split_ratio):
    return f"data_server_{round(server_split_ratio,ndigits=2):.1f},data_use_"+str(percent_train_data_use)+"_with_server_net_"+str(with_server_net)+"_epoch_num_"+str(epochs_num_input)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define VGG16 for server
class VGGServer(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGServer, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=None)  # No pre-trained weights
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)  # Adjust for CIFAR-10

    def forward(self, x):
        # Resize input to match VGG's expected input size
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.vgg(x)
        return x


def get_client_model():
    if client_net_type == NetType.ALEXNET:
        return AlexNet(num_classes=num_classes).to(device)
    if client_net_type == NetType.VGG:
        return VGGServer(num_classes=num_classes).to(device)


def get_server_model():
    if server_net_type == NetType.ALEXNET:
        return AlexNet(num_classes=num_classes).to(device)
    if server_net_type == NetType.VGG:
        return VGGServer(num_classes=num_classes).to(device)




class LearningEntity(ABC):
    def __init__(self,id_,global_data,test_data):
        self.global_data = global_data
        self.pseudo_label_received = None
        self.pseudo_label_to_send = None
        self.current_iteration = 0
        self.epoch_count = 0
        self.id_ = id_
        self.test_set= test_data
        self.model=None
        self.weights = None
    def initialize_weights(self, layer):
        """Initialize weights for the model layers."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def iterate(self,t):
        if self.weights  is None:
            self.model.apply(self.initialize_weights)
        else:
            self.model.apply(self.weights)

        self.iteration_context(t)

    @abstractmethod
    def iteration_context(self,t):
        pass

    def train(self,mean_pseudo_labels, data_):


        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(data_, batch_size=batch_size, shuffle=False, num_workers=4,
                                   drop_last=True)

        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(epochs_num_input):
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
                start_idx = batch_idx * batch_size
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
            print(f"Epoch [{epoch + 1}/{epochs_num_input}], Loss: {avg_loss:.4f}")

        self.weights =self.model.state_dict()
        return avg_loss

    def evaluate(self,data_):
        # Set the model to evaluation mode
        self.model.load_state_dict(self.weights)
        self.model.eval()

        # Create a DataLoader for the global data
        global_loader = DataLoader(data_, batch_size=batch_size, shuffle=False)

        # List to hold pseudo-labels
        pseudo_labels_list = []

        with torch.no_grad():
            for batch in global_loader:
                # Unpack the batch if it contains both inputs and targets
                inputs = batch[0]  # Assuming the first element is the inputs
                inputs = inputs.to(device)  # Move to device

                # Get the model outputs
                outputs = self.model(inputs)

                # Assuming outputs are class probabilities
                pseudo_labels_list.append(outputs.cpu())  # Store on CPU for easier aggregation

        # Concatenate all pseudo labels from the batch
        pseudo_labels = torch.cat(pseudo_labels_list, dim=0)

        # Return pseudo labels to send to the server
        return pseudo_labels  # This should be a tensor of shape (num_samples, num_classes)

    def evaluate_test_loss(self):
        """Evaluate the model on the test set and return the loss."""
        self.model.eval()  # Set the model to evaluation mode
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()  # Define the loss function
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move to device

                # Get the model outputs
                outputs = self.model(inputs)
                # Calculate loss
                loss = criterion(outputs, targets)

                # Accumulate the loss
                total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
                total_samples += inputs.size(0)  # Count the number of samples
        ans = total_loss / total_samples if total_samples > 0 else float('inf')
        print(f"Iteration [{self.current_iteration}], Test Loss: {ans:.4f}")

        # Return average loss
        return ans  # Avoid division by zero

class Client(LearningEntity):
    def __init__(self, id_, client_data, global_data,test_data,class_):
        LearningEntity.__init__(self,id_,global_data,test_data)
        self.local_data = client_data
        self.class_ = class_
        self.epoch_count = 0
        self.model = get_client_model()
        self.weights = None

        self.global_data_list = []
        for data_list in self.global_data.values():
            self.global_data_list.extend(data_list)
        random.seed(seed_num)
        random.shuffle(self.global_data_list)

    def iteration_context(self, t):
        self.current_iteration = t
        if t>0:
            train_loss = self.train(self.pseudo_label_received,self.global_data_list)
        train_loss = self.fine_tune()
        self.pseudo_label_to_send = self.evaluate()
        test_loss = self.evaluate_test_loss()

    def __str__(self):
        return "Client " + str(self.id_)

    def fine_tune(self):
        print("*** " + self.__str__() + " fine-tune ***")

        # Load the weights into the model
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        else:
            self.model.apply(self.initialize_weights)

        # Create a DataLoader for the local data
        fine_tune_loader = DataLoader(self.local_data, batch_size=batch_size, shuffle=True)
        self.model.train()  # Set the model to training mode

        # Define loss function and optimizer

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs_num_input):
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
            self.add_train_loss_to_per_epoch(result_to_print, "Fine Tuning")
            print(f"Epoch [{epoch + 1}/{epochs_num_input}], Loss: {result_to_print:.4f}")

        return  result_to_print


class Server(LearningEntity):
    def __init__(self,id_,global_data,test_data, clients_ids):
        LearningEntity.__init__(self, id_,global_data,test_data)
        self.received_pseudo_labels = {}
        self.clients_ids = clients_ids
        self.reset_clients_received_pl()
        self.model = get_server_model()
        self.weights = self.initialize_weights





    def receive_single_pseudo_label(self, sender, info):
        self.received_pseudo_labels[sender] = info


    def iteration_context(self,t):
        self.current_iteration = t
        self.server_split_ratio = server_split_ratio
        if with_server_net:
            mean_pseudo_labels = self.get_mean_pseudo_labels()  # #
            self.model = get_server_model()
            weights_train,train_loss = self.train(mean_pseudo_labels)
            self.pseudo_label_to_send = self.evaluate(weights_train)
            test_loss = self.evaluate_test_loss()

        else:
            self.pseudo_label_to_send = self.get_mean_pseudo_labels()

        self.reset_clients_received_pl()





    def reset_clients_received_pl(self):
        for id_ in self.clients_ids:
            self.received_pseudo_labels[id_] = None

    def get_mean_pseudo_labels(self):
        # Stack the pseudo labels tensors into a single tensor

        pseudo_labels_list = list(self.received_pseudo_labels.values())

        stacked_labels = torch.stack(pseudo_labels_list)

        # Average the pseudo labels across clients
        average_pseudo_labels = torch.mean(stacked_labels, dim=0)

        return average_pseudo_labels



    def __str__(self):
        return "server"






