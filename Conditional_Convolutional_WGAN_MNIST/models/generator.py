import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.linear0 = nn.Sequential(
            nn.Linear(64,2048,bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU()
        )
        
        self.trans_conv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048,
                               out_channels=512,
                               kernel_size=(4,4),
                               stride=1
                              ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=64,
                               kernel_size=(4,4),
                               stride=2,
                               padding=1
                              ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=(4,4),
                               stride=2,
                               padding=2), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        
        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               kernel_size=(4,4),
                               stride=2,
                               padding=0), 
            nn.LeakyReLU()
        )
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=0,
                      dilation=1),
            
        )



    def forward(self,z):
        x = self.linear0(z)
#         print(x.shape)
        x = x.view(x.shape[0],x.shape[1],1,1)
#         print(x.shape)
        x = self.trans_conv0(x)
#         print(x.shape)
        x = self.trans_conv1(x)
#         print(x.shape)
        x = self.trans_conv2(x)
#         print(x.shape)
        x = self.trans_conv3(x)
#         print(x.shape)
        x = self.conv0(x)
#         print(x.shape)
        return x
