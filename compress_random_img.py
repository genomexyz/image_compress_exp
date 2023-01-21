import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import cv2

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

print(model_load)

#read img data

img = cv2.imread('sampel.jpg')
down_width = 32
down_height = 32
down_points = (down_width, down_height)
resized_down = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)

#save resized sampel
cv2.imwrite('resized_sampel.jpg', resized_down)

img_resized = np.transpose(resized_down)
img1d = np.reshape(img_resized, (-1))
img1d = img1d / 255
img1d_torch = torch.from_numpy(img1d).float()

print(img1d, np.shape(img1d))

output_encode, output_decode = model_load(img1d_torch)
print(output_encode)
torch.save(output_encode, "compress_sampel17.pt")