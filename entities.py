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





class FL_Entity:
    def __init__(self, id_, global_data, test_set, server_split_ratio):
        self.server_data = global_data
        self.test_set = test_set
        self.model = None
        self.pseudo_label_to_send = None
        self.id_ = id_
        self.epoch_count = 0
        self.server_split_ratio = server_split_ratio
        meta_list = list(get_meta_data().keys())
        self.train_df = pd.DataFrame(columns=meta_list+["Id", 'Iteration', 'Loss','Phase','Epoch Count'])
        self.eval_test_df = pd.DataFrame(columns=meta_list+["Id", 'Iteration', 'Train Loss', 'Test Loss'])
        self.current_iteration=0
        self.pseudo_label_for_train = None
        self.batch_size_train= None
        self.learning_rate_train = None
        self.weights = None


    @staticmethod
    def initialize_weights(layer):
        """Initialize weights for the model layers."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def __str__(self):
        return str(self.id_)


    def train(self ):
        import torch.nn.functional as F

        print(f"*** {self.__str__()} train ***")

        server_loader,criterion,optimizer,pseudo_targets_all = self.prep_for_train()
        #check if shuffle cause a bug see loss rate

        # Loop through the specified number of epochs
        for epoch in range(epochs_num_input):

            # Increment the epoch count
            self.epoch_count += 1

            # Initialize the total loss for this epoch
            epoch_loss = 0

            # Iterate through the batches in the server_loader
            for batch_idx, (inputs, _) in enumerate(server_loader):
                # Move input data to the specified device (GPU/CPU)
                inputs = inputs.to(device)

                # Reset the gradients for the optimizer
                optimizer.zero_grad()

                # Forward pass: compute model outputs
                outputs = self.model(inputs)

                # Convert model outputs to log probabilities
                outputs_prob = F.log_softmax(outputs, dim=1)

                # Determine the start and end indices for slicing pseudo targets
                start_idx = batch_idx * self.batch_size_train
                end_idx = start_idx + inputs.size(0)

                # Slice pseudo_targets for the current batch and move to the device
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                # Normalize pseudo targets to sum to 1
                pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # Calculate the loss between model output probabilities and pseudo targets
                loss = criterion(outputs_prob, pseudo_targets)

                # Check if the loss is NaN or Inf (optional but could be useful)
                # if not torch.isfinite(loss): # You could raise an error or log here

                # Backward pass: compute gradients
                loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update the model parameters
                optimizer.step()

                # Accumulate the total loss for the epoch
                epoch_loss += loss.item()

            # Calculate the average loss for the epoch
            avg_loss = epoch_loss / len(server_loader)
            self.add_train_loss_to_per_epoch(avg_loss,"Training")

            # Print the current epoch's average loss
            print(f"Epoch [{epoch + 1}/{epochs_num_input}], Loss: {avg_loss:.4f}")

        return self.model.state_dict(),avg_loss

    def prep_for_train(self):
        server_loader = DataLoader(self.server_data, batch_size=self.batch_size_train, shuffle=True, num_workers=4,
                                   drop_last=True)

        # Initialize model weights
        if with_prev_weights and self.weights is not None:
            self.model.load_state_dict(self.weights)
        else:
            self.model.apply(self.initialize_weights)
        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_train)
        pseudo_targets_all = self.pseudo_label_for_train.to(device)
        return server_loader,criterion,optimizer,pseudo_targets_all


    def evaluate(self):
        print("***"+ self.__str__()+ " evaluation ***")

        # Load the trained weights into the model
        self.model.load_state_dict(self.weights)
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


    def add_train_loss_to_per_epoch(self,result_to_print,phase):
        meta_data_dict = get_meta_data()
        meta_data_df = pd.DataFrame(meta_data_dict)

        new_data = {
            'Sever Data Percentage': [self.server_split_ratio],
            "Id": [str(self.id_)],
            'Iteration': [self.current_iteration],
            'Loss': [result_to_print],
            'Phase': [phase],
            'Epoch Count': [self.epoch_count]
        }

        # Create a new DataFrame from the new data
        new_data_df = pd.DataFrame(new_data)
        current_result = pd.concat([meta_data_df.reset_index(drop=True), new_data_df.reset_index(drop=True)], axis=1)

        # Combine the existing and new DataFrames (if you want to concatenate them)
        #to_add = pd.concat([meta_data_df, new_data_df], ignore_index=True)
        self.train_df = pd.concat([self.train_df, current_result], ignore_index=True)

    def add_to_data_frame(self,test_loss,train_loss):
        # Assuming get_meta_data() returns a DataFrame
        meta_data_dict = get_meta_data()
        meta_data_df = pd.DataFrame(meta_data_dict)
        # Create a dictionary for the new data
        new_data = {
            'Id': [str(self.id_)],
            'Iteration': [self.current_iteration],
            'Train Loss': [float(train_loss)],
            'Test Loss': [float(test_loss)]  # Explicitly convert to float
        }

        # Create a new DataFrame from the new data
        new_data_df = pd.DataFrame(new_data)
        current_result = pd.concat([meta_data_df.reset_index(drop=True), new_data_df.reset_index(drop=True)], axis=1)

        # Combine the existing DataFrame with the new one

        # Update self.eval_test_df with the combined results
        self.eval_test_df = pd.concat([self.eval_test_df, current_result], ignore_index=True)
        #file_name = get_file_name(self.server_split_ratio)+","+self.__str__()+"test_train_loss.csv"
        #self.eval_test_df.to_csv(file_name)
        #plot_loss_per_client(average_loss_df= self.eval_test_df, filename=file_name,client_id = self.id_,server_split_ratio = self.server_split_ratio)

    def evaluate_test_loss(self,train_loss):
        """Evaluate the model on the test set and return the loss."""
        self.model.eval()  # Set the model to evaluation mode
        test_loader = DataLoader(self.test_set, batch_size=client_batch_size_evaluate, shuffle=True)

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
        self.add_to_data_frame(ans, train_loss)

        # Return average loss
        return ans  # Avoid division by zero


class Server(FL_Entity):
    def __init__(self, id_, global_data, test_set, server_split_ratio, clients_ids):


        FL_Entity.__init__(self, id_, global_data, test_set, server_split_ratio)
        self.model = get_server_model()
        self.received_pseudo_labels = {}
        self.clients_ids = clients_ids
        self.reset_clients_received_pl()
        self.batch_size_train= server_batch_size_train

        self.learning_rate_train = server_learning_rate_train
    def receive_single_pseudo_label(self, sender, info):
        self.received_pseudo_labels[sender] = info

    def iterate(self, t):
        self.current_iteration = t
        if with_server_net:
            self.pseudo_label_for_train = self.get_mean_pseudo_labels()  # #
            self.model = get_server_model()
            self.weights,train_loss = self.train()
            self.pseudo_label_to_send = self.evaluate()
            test_loss = self.evaluate_test_loss(train_loss)
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





class Client(FL_Entity):
    def __init__(self, id_, global_data, test_set, server_split_ratio, client_data):
        FL_Entity.__init__(self, id_, global_data, test_set, server_split_ratio)

        self.local_data = client_data
        self.model = None  # get_client_model()
        self.batch_size_train= client_batch_size_train
        self.learning_rate_train = client_learning_rate_train


    def iterate(self, t):
        self.current_iteration = t
        self.model = get_client_model()
        if t>0:
            self.weights,train_loss = self.train()
        self.weights,train_loss = self.fine_tune()
        self.pseudo_label_to_send = self.evaluate()
        test_loss = self.evaluate_test_loss(train_loss)
        #self.add_to_data_frame(test_loss,train_loss)
        #self.handle_data_per_epoch()





    def fine_tune(self):
        print("*** " + self.__str__() + " fine-tune ***")

        # Load the weights into the model
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        else:
            self.model.apply(self.initialize_weights)

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
            self.add_train_loss_to_per_epoch(result_to_print,"Fine Tuning")

            print(f"Epoch [{epoch + 1}/{epochs_num_input}], Loss: {result_to_print:.4f}")

        # Return the model weights after fine-tuning
        return self.model.state_dict(),result_to_print

    def evaluate(self):
        # Set the model to evaluation mode
        self.model.load_state_dict(self.weights)
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






    def handle_data_per_epoch(self):
        # Save the current training data (self.train_df) to a CSV file
        train_file_name = get_file_name(self.server_split_ratio) + "," + self.__str__() + ",train_loss_per_epoch.csv"
        self.train_df.to_csv(train_file_name, index=False)  # Ensure it's saved without the index

        # Plotting the Loss vs Epoch Count
        plt.figure(figsize=(10, 6))  # Set figure size

        # Filter only training phase data
        #train_phase_data = self.train_df[self.train_df['Phase'] == "Train"]

        # Plot training loss over epochs
        plt.plot(self.train_df['Epoch Count'], self.train_df['Loss'], label="Training Loss", marker='o')

        # Add labels and title
        plt.xlabel('Epoch Count')
        plt.ylabel('Loss')
        plt.title(f"Client {self.id_} - Training Loss vs Epoch Count (Server Data {self.server_split_ratio * 100}%)")

        # Add a grid and legend
        plt.grid(True)
        plt.legend()

        # Save the plot to a file
        plot_file_name = get_file_name(
            self.server_split_ratio) + "," + self.__str__() + "_train_loss_per_epoch_plot.png"
        plt.savefig(plot_file_name)
        yyy=3
        # Optionally show the plot (in case you want to view it during runtime)
        #plt.show()

