import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torchvision
from torch.utils.data import  DataLoader
import torchvision.transforms.functional as TF

from PIL import Image
from skimage import io,color
import numpy as np

def prepare_image(file1,file2):
    h,w = 178, 178

    i = io.imread(file1.file)
    m = io.imread(file2.file)

    # image to a Torch tensor 
    transform = transforms.Compose([ 
        transforms.ToTensor(),transforms.Lambda(lambda x: x[:3]),
    transforms.Resize((h,w), antialias=True) 
    ]) 

    img_tensor = transform(i)
    mask_tensor = transform(m)
    to_return = torch.maximum(mask_tensor, img_tensor)


    # pil_img = torchvision.transforms.functional.to_pil_image(img_tensor)
    # pil_img.save("pil_img.png")
    # pil_img = torchvision.transforms.functional.to_pil_image(mask_tensor)
    # pil_img.save("pil_mask.png")
    # merged_img = torchvision.transforms.functional.to_pil_image(torch.maximum(mask_tensor,img_tensor))
    # merged_img.save("merged_Img.png")
    # print(img_tensor)
    # img_tensor = torch.round(img_tensor, decimals=3)
    # mask_tensor = torch.round(mask_tensor, decimals=1)
    # print(mask_tensor)

    # print(img_tensor.shape)
    # print(img_tensor)
    return torch.unsqueeze(to_return, 0)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            
        )

    def forward(self, x):
        return self.conv(x)
    







class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        # down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.ups.append(
                # output = s * (n-1) + k- 2*p
                nn.ConvTranspose2d(
                    feature*2, feature,kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2,feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)
