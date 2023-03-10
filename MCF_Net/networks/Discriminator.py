import torch.nn as nn

# 输出为二分类的概率值，输出使用sigmoid激活 0-1
# BCEloss计算交叉熵损失
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(3072, 512),
            nn.LeakyReLU(),  # f(x) ： x>0输出x，如果x<0输出 a*x a表示一个很小的斜率，比如0.1
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )
    def forward(self,x):
        x = x.view(-1,3072)
        x = self.main(x)
        return x