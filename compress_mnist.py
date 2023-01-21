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
cnt = 0
for data in test_loader:
    data = data[0]
    img_single = data[0]
    data = data.view(data.size(0), -1)
    output_encode, output_decode = model(data)
    torch.save(output_encode, "compression_val_mnist/%s.pt"%(cnt))

    img_numpy = img_single.numpy() * 0.5 + 0.5
    img_numpyT = img_numpy.astype(np.uint8)
    img_numpyT = img_numpyT[0]
    img_numpyT *= 255
    print(img_numpyT, np.shape(img_numpyT))

    img_8bit = Image.fromarray(img_numpyT)
    #print('cek img 8 bit', np.shape(img_numpyT),np.shape(img_numpy))
    img_8bit.save("real_img_mnist/%s.png"%(cnt))

    cnt += 1
    if cnt > cnt_limit:
        break