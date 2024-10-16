import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

# Define the device to use for training (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Training function for clients
def train_client(client_id, model, local_data, global_data, epochs=5, batch_size=32):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate
    criterion = nn.CrossEntropyLoss()

    # Create a combined DataLoader for local and global data
    combined_data = torch.utils.data.ConcatDataset([local_data, global_data])
    data_loader = DataLoader(combined_data, batch_size=batch_size, shuffle=True)
    print(f"------- Client {client_id} Training ------------")

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

        print(f"Client {client_id} - Epoch {epoch + 1}, Loss: {running_loss / len(data_loader):.4f}")

    return model.state_dict()  # Return the model's weights

# Function to create pseudo-labels from predictions
def create_pseudo_labels(model, global_data):
    model.eval()
    pseudo_labels = []
    with torch.no_grad():
        for inputs, _ in global_data:
            inputs = inputs.to(device)
            outputs = model(inputs)
            pseudo_labels.append(outputs.cpu())  # Collect predictions
    return torch.cat(pseudo_labels)

# Aggregation function for server
def aggregate_pseudo_labels(pseudo_labels):
    # Simple averaging of pseudo-labels
    return torch.mean(torch.stack(pseudo_labels), dim=0)

# Function to update the server model with aggregated weights
def update_server_model(server_model, aggregated_weights):
    server_model.load_state_dict(aggregated_weights, strict=False)
    return server_model

if __name__ == '__main__':
    # Load CIFAR-10 dataset with transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(32, padding=4),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Subset for demonstration purposes (e.g., 5% of the dataset)
    dataset_size = len(trainset)
    five_percent_size = int(0.20 * dataset_size)
    trainset = torch.utils.data.Subset(trainset, range(five_percent_size))

    # Distributing the data
    dataset_size = len(trainset)
    client_1_size = int(0.4 * dataset_size)  # 40% for client 1
    client_2_size = int(0.4 * dataset_size)  # 40% for client 2
    server_size = dataset_size - (client_1_size + client_2_size)  # Remaining 20% for the server

    # Perform the split
    client_1_data, client_2_data, server_data = random_split(trainset, [client_1_size, client_2_size, server_size])

    # Initialize server model as VGG16
    server_model = VGGServer(num_classes=10).to(device)  # Use VGG for server and move to device

    num_rounds = 5  # Number of federated training rounds
    for round in range(num_rounds):
        print(f"--- Round {round + 1} ---")

        # Clients update their local model with server weights before training
        client1_model = AlexNet(num_classes=10).to(device)  # Move model to device
        client2_model = AlexNet(num_classes=10).to(device)  # Move model to device

        # Load the server's aggregated model weights into each client's model
        client1_model.load_state_dict(server_model.state_dict(), strict=False)
        client2_model.load_state_dict(server_model.state_dict(), strict=False)

        # Clients train on their local data and global dataset
        client1_weights = train_client(1, client1_model, client_1_data, server_data, epochs=5, batch_size=64)
        client2_weights = train_client(2, client2_model, client_2_data, server_data, epochs=5, batch_size=64)

        # Create pseudo-labels for the global dataset from each client model
        pseudo_labels_client1 = create_pseudo_labels(client1_model, DataLoader(server_data, batch_size=64))
        pseudo_labels_client2 = create_pseudo_labels(client2_model, DataLoader(server_data, batch_size=64))

        # Aggregate pseudo-labels at the server
        aggregated_labels = aggregate_pseudo_labels([pseudo_labels_client1, pseudo_labels_client2])


    print("Federated Learning complete.")
