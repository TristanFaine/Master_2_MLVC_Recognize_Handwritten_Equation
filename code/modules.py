import torch.nn
import torch.nn.functional as F

class SegmentSelector(torch.nn.Module):
    def __init__(self, hiddencells = 100):
        super(SegmentSelector, self).__init__()
        self.fc1 = torch.nn.Linear(32 * 32 , hiddencells)
        self.fc2 = torch.nn.Linear(hiddencells, 2)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x