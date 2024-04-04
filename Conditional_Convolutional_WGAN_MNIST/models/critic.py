import torch.nn as nn

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(11,32,5,2,2),
            nn.LayerNorm([32,14,14]),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32,512,5,2,2),
            nn.LayerNorm([512,7,7]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512,512,5,2,2),
            nn.LayerNorm([512,4,4]),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.linear0 = nn.Linear(512,1)
        
    def forward(self,img):
        x = self.conv0(img)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool(x).flatten(1)
        out = self.linear0(x)
        return out
