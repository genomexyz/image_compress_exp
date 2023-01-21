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

print(model_load)

# Load the image data
transform = transforms.Compose([transforms.ToTensor()])
image_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)

# Define the data loader
data_loader = torch.utils.data.DataLoader(image_data, batch_size=1, shuffle=True)

#for data in data_loader:
#    img, _ = data
#    img = img[0]
#    img_numpy = img.numpy() * 255
#    img_numpyT = np.transpose(img_numpy)
#    img_numpyT = img_numpyT.astype(np.uint8)
#    print('cek min', np.min(img_numpyT))
#    print('cek max', np.max(img_numpyT))
#    #print(img_numpyT)
#    img_8bit = Image.fromarray(img_numpyT)
#    img_8bit.save("img.png")
#    break

cnt = 0
cnt_limit = 100
for data in data_loader:
    img, _ = data
    img_single = img[0]
    img = img.view(img.size(0), -1)
 
    img_numpy = img_single.numpy() * 255
    img_numpyT = np.transpose(img_numpy)
    img_numpyT = img_numpyT.astype(np.uint8)

    img_8bit = Image.fromarray(img_numpyT)
    #print('cek img 8 bit', np.shape(img_numpyT),np.shape(img_numpy))
    img_8bit.save("real_img17/%s.png"%(cnt))

    output_encode, output_decode = model_load(img)
    print(output_encode)
    torch.save(output_encode, "compression_val17/%s.pt"%(cnt))

    cnt += 1
    if cnt > cnt_limit:
        break