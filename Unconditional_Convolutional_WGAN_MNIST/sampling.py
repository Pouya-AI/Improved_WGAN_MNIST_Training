import torch
import matplotlib.pyplot as plt
from Models.Generator import Generator

gen = Generator('cpu').eval()
gen_dir = 'Models/Saved_State_Dicts/Unconditional_Convolutional_Generator_state_dict.pth'
state_dict = torch.load(gen_dir)
gen.load_state_dict(state_dict)

z = torch.randn(400,32)
with torch.no_grad():    
    imgs = gen(z)
fig,axes = plt.subplots(20,20,figsize=(80,80))
iteration = 0
for i in range(len(axes)):
    for j in range(len(axes)):
        axes[i][j].axis('off')
        axes[i][j].imshow(imgs[iteration].permute(1,2,0),'gray')
        iteration += 1