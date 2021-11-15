import torch
import torch.nn as nn

class mm_cls(nn.Module):
    def __init__(self, backbone, out_channels=1000, num_classes=2, embed_dim=0):
        super().__init__()

        self.backbone = backbone
        self.embed_dim = embed_dim

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(out_channels+embed_dim, num_classes)

    def forward(self, x):
        embedding = None
        if type(x) is list:
            x, embedding = x

        x = self.backbone(x)

        if self.embed_dim > 0:
            x = torch.cat([x,embedding], 1)

        x = self.dropout(x)
        x = self.fc(x)

        return x


class levi(nn.Module):
    def __init__(self, inplanes=3):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, 96, kernel_size=7, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding='same')

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.norm = nn.LocalResponseNorm(5)
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(384*6*6, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class levi_bn(nn.Module):
    def __init__(self, inplanes=3):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, 96, kernel_size=7, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding='same')
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding='same')

        self.norm1 = nn.BatchNorm2d(96)
        self.norm2 = nn.BatchNorm2d(256)
        self.norm3 = nn.BatchNorm2d(384)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(384*6*6, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class levi_2layer(nn.Module):
    def __init__(self, inplanes=3):
        super().__init__()

        self.conv1 = nn.Conv2d(inplans, 96, kernel_size=7, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding='same')

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.norm = nn.LocalResponseNorm(5)
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(256*13*13, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
