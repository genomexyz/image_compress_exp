import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

class Autoencoder(nn.Module):
    def __init__(self,  input_size, compress_total):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, compress_total))  #<-- change the last layer to have 10 neurons
        
        self.decoder = nn.Sequential(
            nn.Linear(compress_total, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Tanh())

    def forward(self, x):
        #print('cek x input', x.size())
        x_encoder = self.encoder(x)
        #print('cek x', x.size())
        x_decoder = self.decoder(x_encoder)
        #print('cek x decoder', x.size())
        return x_encoder, x_decoder

# Create an instance of the autoencoder
model = Autoencoder(28*28, 10)
model.load_state_dict(torch.load('mnist_autoencoder.pt', map_location=torch.device('cpu')))

# Define the transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST test dataset
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)

cnt_limit = 100
for i in range(cnt_limit+1):
    sample_torch = torch.load('compression_val_mnist/%s.pt'%(i))
    decompressed_image_raw = model.decoder(sample_torch)[0]
    decompressed_image_numpy = decompressed_image_raw.detach().numpy() * 0.5 + 0.5
    #decompressed_image_numpy = np.transpose(decompressed_image_numpy)
    image_numpy_2d = np.reshape(decompressed_image_numpy, (28, 28))
    image_numpy_2d = np.transpose(image_numpy_2d)
    img8bit = image_numpy_2d.astype(np.uint8)

    img_8bit = Image.fromarray(img8bit)
    img_8bit.save("compression_res_mnist/%s.png"%(i))