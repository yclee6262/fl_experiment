# filename: models.py
import torch
import torch.nn as nn
from medmnist import INFO

# 設定使用 PathMNIST (結腸病理切片分類，9個類別)
DATA_FLAG = 'pathmnist'
info = INFO[DATA_FLAG]
N_CHANNELS = info['n_channels']
N_CLASSES = len(info['label'])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, N_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")