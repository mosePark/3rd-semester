'''
합성곱 연산이란 커널(kernel) 또는 필터(filter) 라는 n × m 크기의 행렬로
높이(height) × 너비(width) 크기의 이미지를 처음부터 끝까지 겹치며 훑으면서
n × m 크기의 겹쳐지는 부분의 각 이미지와 커널의 원소의 값을 곱해서 모두 더한 값을 출력으로 하는 것을 말합니다.
이때, 이미지의 가장 왼쪽 위부터 가장 오른쪽 아래까지 순차적으로 훑습니다.


[ref](https://wikidocs.net/80437)
'''


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
