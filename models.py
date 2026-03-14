import torch
import torch.nn as nn

# 預設值設為 PathMNIST 的規格 (3 channel, 9 classes) 以保持兼容性

class Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        """
        參數:
        - in_channels: 輸入圖片的通道數 (PathMNIST/BloodMNIST 都是 3)
        - num_classes: 分類輸出的數量 (PathMNIST=9, BloodMNIST=8)
        """
        super(Net, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 這裡假設輸入圖片大小固定為 28x28
        # 經過兩次 MaxPool2d(2) 後，大小變為 7x7 (28 -> 14 -> 7)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")