import threading
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *


# Define AlexNet for clients


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


class Server:
    def __init__(self, global_data, clients_ids):
        self.server_data = global_data
        self.model = get_server_model()
        self.received_pseudo_labels = {}
        self.clients_ids = clients_ids
        self.pseudo_label_to_send = None
        self.reset_clients_ids()

    def receive_single_pseudo_label(self, sender, info):
        self.received_pseudo_labels[sender] = info

    def iterate(self):
        option1 = True
        if option1:
            self.pseudo_label_to_send = self.get_mean_pseudo_labels()
        else:
            mean_pseudo_labels = self.get_mean_pseudo_labels()# #
            self.model = get_server_model()
            weights_train = self.train(mean_pseudo_labels)
            self.pseudo_label_to_send = self.evaluate(weights_train)
        self.reset_clients_ids()

    def reset_clients_ids(self):
        for id_ in self.clients_ids:
            self.received_pseudo_labels[id_] = None

    def get_mean_pseudo_labels(self):
        # Stack the pseudo labels tensors into a single tensor

        pseudo_labels_list = list(self.received_pseudo_labels.values())

        stacked_labels = torch.stack(pseudo_labels_list)

        # Average the pseudo labels across clients
        average_pseudo_labels = torch.mean(stacked_labels, dim=0)

        return average_pseudo_labels

    def train(self,mean_pseudo_labels):
        print("*** " + "server" + " train ***")

        # Calculate the exact batch size to evenly divide the data size
        data_size = len(self.server_data)
        batch_size = server_batch_size_train

        # Adjust batch size if it doesn't perfectly divide the data size
        if data_size % batch_size != 0:
            batch_size = data_size // (data_size // batch_size)

        # Create a DataLoader for the local data
        server_loader = DataLoader(self.server_data, batch_size=batch_size, shuffle=False, num_workers=4)

        # Set the model to training mode
        self.model.train()

        # Define your loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Assuming a classification task
        optimizer = torch.optim.Adam(self.model.parameters(), lr=server_learning_rate_train)

        # Convert pseudo labels to class indices if they're not already
        pseudo_targets_all = torch.argmax(mean_pseudo_labels, dim=1).to(device)

        for epoch in range(server_epochs_train):  # Loop over the number of epochs
            epoch_loss = 0  # Track the loss for this epoch
            start_idx = 0  # Initialize index for slicing pseudo labels
            end_idx = 0  # Initialize end_idx to avoid UnboundLocalError

            for inputs, targets in server_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move to device
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = self.model(inputs)

                # Determine the target labels

                # Use pseudo labels, and slice them to match the input batch size
                end_idx = start_idx + inputs.size(0)  # Calculate the end index for slicing
                pseudo_targets = pseudo_targets_all[start_idx:end_idx]  # Slice to match batch size

                if pseudo_targets.size(0) != inputs.size(0):
                    raise ValueError(
                        f"Pseudo target size {pseudo_targets.size(0)} does not match input size {inputs.size(0)}")

                loss = criterion(outputs, pseudo_targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += loss.item()

                # Update start index for next batch
                start_idx = end_idx

            # Print epoch loss or do any other logging if needed
            print(f"Epoch [{epoch + 1}/{server_epochs_train}], Loss: {epoch_loss / len(server_loader):.4f}")

        # Return the model weights after training
        return self.model.state_dict()  # Return the model weights as a dictionary

    def evaluate(self, weights_train):
        print("*** Server evaluation ***")

        # Load the trained weights into the model
        self.model.load_state_dict(weights_train)
        self.model.eval()  # Set the model to evaluation mode

        # Create a DataLoader for the global data
        global_loader = DataLoader(self.server_data, batch_size=server_batch_size_evaluate, shuffle=False)

        # List to hold pseudo labels
        pseudo_labels_list = []

        with torch.no_grad():  # Disable gradient computation
            for inputs, _ in global_loader:
                inputs = inputs.to(device)  # Move to device

                # Forward pass
                outputs = self.model(inputs)

                # Assuming outputs are class probabilities (soft labels)
                pseudo_labels_list.append(outputs.cpu())  # Store the pseudo labels on the CPU for easier aggregation

        # Concatenate all pseudo labels from the batches
        pseudo_labels = torch.cat(pseudo_labels_list, dim=0)

        # Return the pseudo labels (shape: num_samples x num_classes)
        return pseudo_labels


class Client:
    def __init__(self, id_, client_data, server_data,test_data):
        self.id_ = id_
        self.local_data = client_data
        self.server_data = server_data
        self.model = None  # get_client_model()
        self.pseudo_label_received = None
        self.pseudo_label_to_send = None
        self.current_iteration = 0
        self.test_set = test_data
        self.results_df = pd.DataFrame(columns=['Client','Iteration', 'Test Loss'])

    def iterate(self, t):
        self.current_iteration = t
        self.model = get_client_model()
        train_weights = self.train()
        fine_tune_weights = self.fine_tune(train_weights)
        self.pseudo_label_to_send = self.evaluate(fine_tune_weights)
        test_loss = self.evaluate_test_loss()
        print(f"Iteration [{self.current_iteration}], Test Loss: {test_loss:.4f}")
        current_result = pd.DataFrame({'Client':["c"+str(self.id_)],'Iteration': [self.current_iteration], 'Test Loss': [test_loss]})
        if not current_result.empty and not current_result.isna().all().any():
            self.results_df = pd.concat([self.results_df, current_result], ignore_index=True)

    def __str__(self):
        return "Client " + str(self.id_)

    def train(self):
        print("*** " + self.__str__() + " train ***")

        # Calculate the exact batch size to evenly divide the data size
        data_size = len(self.server_data)
        batch_size = client_batch_size_train

        # Adjust batch size if it doesn't perfectly divide the data size
        if data_size % batch_size != 0:
            batch_size = data_size // (data_size // batch_size)

        # Create a DataLoader for the local data
        server_loader = DataLoader(self.server_data, batch_size=batch_size, shuffle=False, num_workers=4)

        # Set the model to training mode
        self.model.train()

        # Define your loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Assuming a classification task
        optimizer = torch.optim.Adam(self.model.parameters(), lr=client_learning_rate_train)

        # Convert pseudo labels to class indices if they're not already
        if self.pseudo_label_received is not None:
            pseudo_targets_all = torch.argmax(self.pseudo_label_received, dim=1).to(device)

        for epoch in range(client_epochs_train):  # Loop over the number of epochs
            epoch_loss = 0  # Track the loss for this epoch
            start_idx = 0  # Initialize index for slicing pseudo labels
            end_idx = 0  # Initialize end_idx to avoid UnboundLocalError

            for inputs, targets in server_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move to device
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = self.model(inputs)

                # Determine the target labels
                if self.pseudo_label_received is None:
                    # Use the true targets from local data
                    loss = criterion(outputs, targets)
                else:
                    # Use pseudo labels, and slice them to match the input batch size
                    end_idx = start_idx + inputs.size(0)  # Calculate the end index for slicing
                    pseudo_targets = pseudo_targets_all[start_idx:end_idx]  # Slice to match batch size

                    if pseudo_targets.size(0) != inputs.size(0):
                        raise ValueError(
                            f"Pseudo target size {pseudo_targets.size(0)} does not match input size {inputs.size(0)}")

                    loss = criterion(outputs, pseudo_targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += loss.item()

                # Update start index for next batch
                start_idx = end_idx

            # Print epoch loss or do any other logging if needed
            print(f"Epoch [{epoch + 1}/{client_epochs_train}], Loss: {epoch_loss / len(server_loader):.4f}")

        # Return the model weights after training
        return self.model.state_dict()  # Return the model weights as a dictionary

    def fine_tune(self, train_weights):
        print("*** " + self.__str__() + " fine-tune ***")

        # Load the weights into the model
        self.model.load_state_dict(train_weights)
        # Create a DataLoader for the local data
        fine_tune_loader = DataLoader(self.local_data, batch_size=client_batch_size_fine_tune, shuffle=True)
        # Set the model to training mode
        self.model.train()
        # Define your loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Assuming a classification task
        optimizer = torch.optim.Adam(self.model.parameters(), lr=client_learning_rate_fine_tune)


        for epoch in range(client_epochs_fine_tune):  # You can define separate epochs for fine-tuning if needed
            epoch_loss = 0
            for inputs, targets in fine_tune_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move to device
                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                # Accumulate loss for this epoch
                epoch_loss += loss.item()


            print(f"Epoch [{epoch + 1}/{client_epochs_fine_tune}], Loss: {epoch_loss / len(fine_tune_loader):.4f}")

        # Return the model weights after fine-tuning
        return self.model.state_dict()

    def evaluate(self, fine_tune_weights):
        # Set the model to evaluation mode
        self.model.load_state_dict(fine_tune_weights)
        self.model.eval()

        # Create a DataLoader for the global data
        global_loader = DataLoader(self.server_data, batch_size=client_batch_size_evaluate, shuffle=False)

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
        test_loader = DataLoader(self.test_set, batch_size=client_batch_size_evaluate, shuffle=False)

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

        # Return average loss
        return total_loss / total_samples if total_samples > 0 else float('inf')  # Avoid division by zero