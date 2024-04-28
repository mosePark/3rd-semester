import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 합성곱 계층
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=2)
        # 풀링 계층
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 완전 연결 계층
        # Conv2d와 MaxPool2d를 거친 후의 크기 계산 필요: (28-5+2*2)/1 + 1 = 28 -> 28/2 = 14 (풀링)
        # 다시 한 번 풀링하면: 14/2 = 7
        self.fc = nn.Linear(in_features=10 * 7 * 7, out_features=10)

    def forward(self, x):
        # Conv + ReLU + Max Pool
        x = self.pool(F.relu(self.conv1(x)))
        # 특징 맵을 펼침
        x = x.view(-1, 10 * 7 * 7)
        # 완전 연결 계층
        x = self.fc(x)
        return x

# 모델 인스턴스 생성
model = SimpleCNN()
print(model)
