import torch
import torch.nn as nn

#this project folows Aladdin Persson

architecture_configs = [
    (7, 64, 2, 3), #(kernel_size, no_filters, stride, padding)
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], #(tuple, tuple, times_of_repeat)
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
    def __init__(self, in_channels, out_channels,**kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        convolution = self.conv(x)
        batch_normalization = self.batch_norm(convolution)
        output = self.activation(batch_normalization)
        return output

class YOLOV1(nn.Module):
    def __init__(self,  grids, num_boxes, num_classes, architecture_config, in_channels = 3) :
        super(YOLOV1, self).__init__()
        self.architecture_config = architecture_config
        self.in_channels = in_channels
        self.grids = grids
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darknet = self._create_darknet(self.architecture_config)
        self.fully_connected = self._create_fully_connected(self.grids, self.num_boxes, self.num_classes)

    def forward(self, x):
        darknet = self.darknet(x)
        flatten = torch.flatten(darknet,start_dim = 1)
        fully_connected = self.fully_connected(flatten)

        return fully_connected

    def _create_darknet(self, architecture_config):
        layers = []
        in_channels = self.in_channels

        for x in architecture_config:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size = x[0], stride = x[2], padding = x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            elif type(x) == list:
                for i in range(x[2]):
                    layers += [CNNBlock(in_channels, x[0][1], kernel_size = x[0][0],stride = x[0][2], padding = x[0][3])]
                    layers += [CNNBlock(x[0][1], x[1][1], kernel_size= x[1][0], stride= x[1][2], padding= x[1][3])]
                    in_channels = x[1][1]

        return nn.Sequential(*layers)

    def _create_fully_connected(self, grids, num_boxes, num_classes):
        split_size, boxes, classes = grids, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*split_size*split_size, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, split_size * split_size * (boxes * 5 + classes))
        )


def test_model(s= 7 , num_boxes = 2 , num_classes = 20):
    model = YOLOV1( grids=s, num_boxes=num_boxes, num_classes = num_classes,architecture_config= architecture_configs, in_channels=3)
    x = torch.rand((2,3,448,488))
    print(model(x).shape)

test_model()
