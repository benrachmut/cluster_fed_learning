import threading
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *


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







class Server:
    def __init__(self, global_data, clients_ids,test_data):
        self.server_data = global_data
        self.model = get_server_model()
        self.received_pseudo_labels = {}
        self.clients_ids = clients_ids
        self.pseudo_label_to_send = None
        self.reset_clients_received_pl()
        self.train_df = pd.DataFrame(columns=['Sever Data Percentage', 'Client', 'Iteration', 'epoch', 'Loss','Phase','Epoch Count'])
        self.eval_test_df = pd.DataFrame(columns=['Sever Data Percentage', 'Client', 'Iteration', 'Train Loss', 'Test Loss'])

        self.epoch_count = 0
        self.id_ = "server"
        self.test_set= test_data



    def receive_single_pseudo_label(self, sender, info):
        self.received_pseudo_labels[sender] = info

    def iterate(self, t,client_split_ratio):
        self.current_iteration = t
        self.server_split_ratio = round(1-client_split_ratio,2)
        if with_server_net:
            mean_pseudo_labels = self.get_mean_pseudo_labels()  # #
            self.model = get_server_model()
            weights_train,train_loss = self.train(mean_pseudo_labels)
            self.pseudo_label_to_send = self.evaluate(weights_train)
            test_loss = self.evaluate_test_loss()

            self.add_to_data_frame(test_loss, train_loss)
            self.handle_data_per_epoch()
        else:
            self.pseudo_label_to_send = self.get_mean_pseudo_labels()



        self.reset_clients_received_pl()



    def add_to_data_frame(self,test_loss,train_loss):
        current_result = pd.DataFrame({
            'Sever Data Percentage': self.server_split_ratio,
            'Client': ["c" + str(self.id_)],
            'Iteration': [self.current_iteration],
            'Train Loss': [float(train_loss)],
            'Test Loss': [float(test_loss)]  # Explicitly convert to float
        })
        self.eval_test_df = pd.concat([self.eval_test_df, current_result], ignore_index=True)
        file_name = get_file_name(self.server_split_ratio)+","+self.__str__()+"test_train_loss.csv"
        self.eval_test_df.to_csv(file_name)
        #plot_loss_per_client(average_loss_df= self.eval_test_df, filename=file_name,client_id = self.id_,server_split_ratio = self.server_split_ratio)

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
        ans = total_loss / total_samples if total_samples > 0 else float('inf')
        print(f"Iteration [{self.current_iteration}], Test Loss: {ans:.4f}")

        # Return average loss
        return ans  # Avoid division by zero

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

    def initialize_weights(self, layer):
        """Initialize weights for the model layers."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def __str__(self):
        return "server"

    def train(self,mean_pseudo_labels):
        import torch.nn.functional as F

        print(f"*** {self.__str__()} train ***")


        data_size = len(self.server_data)
        batch_size = client_batch_size_train

        # Create a DataLoader for the local data
        server_loader = DataLoader(self.server_data, batch_size=batch_size, shuffle=False, num_workers=4,
                                   drop_last=True)

        # Initialize model weights
        self.model.apply(self.initialize_weights)

        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=server_learning_rate_train)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(epochs_num_input):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                # Check for NaN or Inf in inputs
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"NaN or Inf found in inputs at batch {batch_idx}: {inputs}")
                    continue

                outputs = self.model(inputs)

                # Check for NaN or Inf in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"NaN or Inf found in model outputs at batch {batch_idx}: {outputs}")
                    continue

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

        return self.model.state_dict(),avg_loss

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


    def add_train_loss_to_per_epoch(self,result_to_print, epoch,phase):
        # Create a new row with the current train loss and epoch details
        current_train_loss = pd.DataFrame({
            'Sever Data Percentage': [self.server_split_ratio],  # You might need to pass or save client_split_ratio
            'Client': ["c" + str(self.id_)],
            'Iteration': [self.current_iteration],
            'epoch': [epoch],
            'Loss': [result_to_print],  # This is the loss for the current epoch
            'Phase': [phase],  # Indicate this is from the training phase
            'Epoch Count':[self.epoch_count]
        })

        # Append the new row to the train_df
        self.train_df = pd.concat([self.train_df, current_train_loss], ignore_index=True)

    def handle_data_per_epoch(self):
        # Save the current training data (self.train_df) to a CSV file
        train_file_name = get_file_name(self.server_split_ratio) + "," + self.__str__() + ",train_loss_per_epoch.csv"
        self.train_df.to_csv(train_file_name, index=False)  # Ensure it's saved without the index



class Client:
    def __init__(self, id_, client_data, server_data,test_data,class_):
        self.id_ = id_
        self.local_data = client_data
        self.server_data = server_data
        self.model = None  # get_client_model()
        self.pseudo_label_received = None
        self.pseudo_label_to_send = None
        self.current_iteration = 0
        self.test_set = test_data
        self.class_ = class_
        self.epoch_count = 0
        self.train_weights = None


    def iterate(self, t):
        self.current_iteration = t
        self.model = get_client_model()
        #train_weights = None

        if t>0:
            self.train_weights = self.train()
        fine_tune_weights,train_loss = self.fine_tune(self.train_weights)
        self.pseudo_label_to_send = self.evaluate(fine_tune_weights)
        test_loss = self.evaluate_test_loss()

    def __str__(self):
        return "Client " + str(self.id_)



    def train(self):
        import torch.nn.functional as F

        print(f"*** {self.__str__()} train ***")
        if self.current_iteration == 0:
            raise RuntimeError("Client should train only at iteration > 0")

        data_size = len(self.server_data)
        batch_size = client_batch_size_train

        # Create a DataLoader for the local data
        server_loader = DataLoader(self.server_data, batch_size=batch_size, shuffle=False, num_workers=4,
                                   drop_last=True)

        # Initialize model weights
        self.model.apply(self.initialize_weights)

        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=client_learning_rate_train)

        pseudo_targets_all = self.pseudo_label_received.to(device)

        for epoch in range(epochs_num_input):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                # Check for NaN or Inf in inputs
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"NaN or Inf found in inputs at batch {batch_idx}: {inputs}")
                    continue

                outputs = self.model(inputs)

                # Check for NaN or Inf in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"NaN or Inf found in model outputs at batch {batch_idx}: {outputs}")
                    continue

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

        self.train_weights = self.model.state_dict()

    def initialize_weights(self, layer):
        """Initialize weights for the model layers."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def fine_tune(self, train_weights):
        print("*** " + self.__str__() + " fine-tune ***")

        # Load the weights into the model
        if train_weights is not None:
            self.model.load_state_dict(train_weights)
        # Create a DataLoader for the local data
        fine_tune_loader = DataLoader(self.local_data, batch_size=client_batch_size_fine_tune, shuffle=True)
        # Set the model to training mode
        self.model.train()
        # Define your loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Assuming a classification task
        optimizer = torch.optim.Adam(self.model.parameters(), lr=client_learning_rate_fine_tune)


        for epoch in range(epochs_num_input):  # You can define separate epochs for fine-tuning if needed
            self.epoch_count = self.epoch_count+1
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

            result_to_print = epoch_loss / len(fine_tune_loader)


            print(f"Epoch [{epoch + 1}/{epochs_num_input}], Loss: {result_to_print:.4f}")

        # Return the model weights after fine-tuning
        return self.model.state_dict(),result_to_print

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
        ans = total_loss / total_samples if total_samples > 0 else float('inf')
        print(f"Iteration [{self.current_iteration}], Test Loss: {ans:.4f}")

        # Return average loss
        return ans  # Avoid division by zero


