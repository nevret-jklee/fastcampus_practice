import torch
import torch.nn as nn

# Architecture가 정의된 클래스
# nn.Module이라는 파이토치의 추상 클래스를 상속받아 나만의 클래스를 정의
# 이 때, 2가지 함수만 override 해줘도 잘 동작한다.
# 1. __init__()
# 2. forward()
class ImageClassifier(nn.Module):    
    def __init__(self,
                 input_size,          # 784
                 output_size):        # 10
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layers = nn.Sequential(     # self.layers라는 객체로 할당
            nn.Linear(input_size, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1),      
            # (batch_size, hidden_size)에서 dim=-1을 지정해줌으로써 hidden_size만 softmax를 수행해주기 위해
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        
        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y
        