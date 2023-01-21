import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10))  #<-- change the last layer to have 10 neurons
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create an instance of the autoencoder
model = Autoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Load the image data
transform = transforms.Compose([transforms.ToTensor()])
image_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)

# Define the data loader
data_loader = torch.utils.data.DataLoader(image_data, batch_size=64, shuffle=True)

# Train the autoencoder
for epoch in range(10):
    for data in data_loader:
        img, _ = data
        # Flatten the image
        img = img.view(img.size(0), -1)
        # Normalize the image
        img = img/255
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(img)
        # Compute the loss
        loss = criterion(output, img)
        # Backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 10, loss.item()))