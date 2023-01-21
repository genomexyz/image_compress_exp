import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

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

model_load = Autoencoder(32*32*3, 17)
model_load.load_state_dict(torch.load('model17.pth', map_location=torch.device('cpu')))

sample_torch = torch.load('compress_sampel17.pt')
decompressed_image_raw = model_load.decoder(sample_torch)
decompressed_image_numpy = decompressed_image_raw.detach().numpy() * 255
#decompressed_image_numpy = np.transpose(decompressed_image_numpy)
image_numpy_2d = np.reshape(decompressed_image_numpy, (3, 32, 32))
image_numpy_2d = np.transpose(image_numpy_2d)
img8bit = image_numpy_2d.astype(np.uint8)
img_8bit = Image.fromarray(img8bit)
img_8bit.save("compress_decompress_sampel17.png")
