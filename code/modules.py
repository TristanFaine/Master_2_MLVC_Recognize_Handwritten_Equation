import torch.nn.functional as F
from torch.nn import Module, Conv2d, MaxPool2d, Linear, BatchNorm2d, Sequential, ReLU, Dropout

class SegmentSelector(Module):
    def __init__(self, hiddencells = 100):
        super(SegmentSelector, self).__init__()
        self.conv1 = Conv2d(1, 6, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


class AlexNet(Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = Sequential(
            Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            BatchNorm2d(96),
            ReLU()
            )
        self.layer2 = Sequential(
            Conv2d(96, 384, kernel_size=5, stride=1, padding=2),
            BatchNorm2d(384),
            ReLU()
            )
        self.layer3 = Sequential(
            Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(384),
            ReLU())
        self.layer4 = Sequential(
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(384),
            ReLU())
        self.layer5 = Sequential(
            Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU()
            )
        self.fc = Sequential(
            Dropout(0.5),
            Linear(9216, 4096),
            ReLU())
        self.fc1 = Sequential(
            Dropout(0.5),
            Linear(4096, 4096),
            ReLU())
        self.fc2= Sequential(
            Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out