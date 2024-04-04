import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.generator import Generator

device = 'cuda'
gen = Generator().eval().to(device)
state_dict = torch.load('models/state_dicts_saved/iteration_2/epoch_100/Conditional_Convolutional_Generator.pth')
gen.load_state_dict(state_dict)

z = torch.randn(100,64).to(device)
z[:,-10:] = 0
for i in range(1,101):
    z[i-1,-(-(i)%10)-1] = 1

with torch.no_grad():
    img = gen(z).squeeze(0)

img = img.to('cpu')

fig,axes = plt.subplots(10,10,figsize=(10,10))
num_iter = 0
for i in range(len(axes)):
    for j in range(len(axes[i])):
        axes[i][j].axis('off')
        axes[i][j].imshow(img[num_iter].permute(1,2,0),'gray')
        num_iter += 1

plt.show()