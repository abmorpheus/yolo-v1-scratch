import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3), # (kernel size, # filters, stride, padding)
    "M", 
    (3, 192, 1, 1), 
    "M", 
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M", 
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], #[layers, repetition]
    (1, 512, 1, 0), 
    (3, 1024, 1, 1), 
    "M", 
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], 
    (3, 1024, 1, 1), 
    (3, 1024, 2, 1), 
    (3, 1024, 1, 1), 
    (3, 1024, 1, 1), 
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              bias = False, kernel_size = kernel_size, 
                              stride = stride, padding = padding
                              )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    

class Yolov1(nn.Module):
    def __init__(self, in_channels, split_size, num_boxes, num_classes):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fullycon = self._create_fullycon(split_size, num_boxes, num_classes)

    def forward(self, x):
        x = self.darknet(x)
        return self.fullycon(torch.flatten(x, start_dim = 1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, out_channels = x[1], kernel_size = x[0], stride = x[2], padding = x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                rep = x[2]
                for _ in range(rep):
                    layers += [CNNBlock(in_channels, out_channels = conv1[1], kernel_size = conv1[0], stride = conv1[2], padding = conv1[3])]
                    layers += [CNNBlock(conv1[1], out_channels = conv2[1], kernel_size = conv2[0], stride = conv2[2], padding = conv2[3])]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)
    
    def _create_fullycon(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(), 
            nn.Linear(S * S * 1024, 496), # original paper = 4096
            nn.Dropout(0.1), 
            nn.LeakyReLU(0.1), 
            nn.Linear(496, S * S * (C + B * 5)),
        )
    

def test(split_size = 7, num_boxes = 2, num_classes = 20):
    model = Yolov1(3, split_size, num_boxes, num_classes)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

test()

