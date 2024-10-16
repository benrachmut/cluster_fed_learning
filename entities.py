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
    def __init__(self,global_data):
        self.global_data = global_data
        self.model = get_server_model()



class Client:
    def __init__(self,id_,client_data,server_data):
        self.id_ = id_
        self.local_data = client_data
        self.server_data = server_data
        self.model = None # get_client_model()
        self.server_pseudo_label = None

        self.current_iteration = 0

    def iterate(self,t):
        self.current_iteration =t
        train_weights = self.train()
        fine_tune_weights = self.fine_tune(train_weights)


    def __str__(self):
        return "Client "+str(self.id_)

    def train(self):
        self.model = get_client_model()

        print("*** "+self.__str__()+ " train ***")
        # Create a DataLoader for the local data
        server_loader = DataLoader(self.local_data, batch_size=client_batch_size_train, shuffle=True)

        # Set the model to training mode
        self.model.train()

        # Define your loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Assuming a classification task
        optimizer = torch.optim.Adam(self.model.parameters(), lr=client_learning_rate_train)

        for epoch in range(client_epochs_train):  # Loop over the number of epochs
            epoch_loss = 0  # Track the loss for this epoch
            for inputs, targets in server_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move to device
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = self.model(inputs)

                # Determine the target labels
                if self.server_pseudo_label is None:
                    # Use the true targets from local data
                    loss = criterion(outputs, targets)
                else:
                    # Use server pseudo labels as ground truth
                    pseudo_targets = self.server_pseudo_label  # Assume this matches the batch size
                    loss = criterion(outputs, pseudo_targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss for this epoch
                epoch_loss += loss.item()

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