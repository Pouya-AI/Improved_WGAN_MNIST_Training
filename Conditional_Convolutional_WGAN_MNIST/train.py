import torch
from utils import *
from torchvision.datasets import MNIST
from torch.utils.data import ConcatDataset
import torchvision.transforms as T
from models.critic import Critic
from models.generator import Generator
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


transform = T.ToTensor()
dataset_train = MNIST(root='datasets/mnist_train',train=True,
                    transform=transform,target_transform=one_hot,download=False)
dataset_test = MNIST(root='datasets/MNIST_test',train=False,
                    transform=transform,target_transform=one_hot,download=False)
concatenated_dataset = ConcatDataset([dataset_train,dataset_test])


def train_critic(args,imgs,noise,gen_lbls,critic_lbls):
    
    args['critic']['model'].train()
    args['gen']['model'].eval()
    
    gen_input = torch.cat((noise,gen_lbls),dim=1)
    with torch.no_grad():
        gen_output = args['gen']['model'](gen_input)
    
    critic_input_gen = torch.cat((gen_output,critic_lbls),dim=1)
    critic_input = torch.cat((imgs,critic_lbls),dim=1)
    
    critic_output_gen = args['critic']['model'](critic_input_gen)
    critic_output = args['critic']['model'](critic_input)
    
    wloss_value = wloss(critic_output,critic_output_gen)
    reg_value = reg(args,critic_input,critic_input_gen)
    loss_value = wloss_value + reg_value
    loss_value.backward()
    
    args['critic']['optim']['algorithm'].step()
    args['critic']['optim']['algorithm'].zero_grad()


def train_gen(args,noise,gen_lbls,critic_lbls):
    
    args['critic']['model'].eval()
    args['gen']['model'].train()
    
    gen_input = torch.cat((noise,gen_lbls),dim=1)
    gen_output = args['gen']['model'](gen_input)
    
    critic_input_gen = torch.cat((gen_output,critic_lbls),dim=1)
    critic_output_gen = args['critic']['model'](critic_input_gen)
    
    loss_value = -1*torch.mean(critic_output_gen)
    loss_value.backward()
    
    args['gen']['optim']['algorithm'].step()
    args['gen']['optim']['algorithm'].zero_grad()
    args['critic']['model'].zero_grad()
    
    
def train_epoch(args):
    
    batch_size = args['data']['batch_size']
    z_features = args['data']['z_features']
    ncritic = args['critic']['ncritic']
    
    num_iter = 0
    for imgs,(gen_lbls,critic_lbls) in tqdm(args['data']['loader']):
        num_iter+=1
        
        imgs = imgs.to(args['device'])
        gen_lbls = gen_lbls.to(args['device'])
        critic_lbls = critic_lbls.to(args['device'])
        noise = torch.randn(batch_size,z_features).to(args['device'])
        
        train_critic(args,imgs,noise,gen_lbls,critic_lbls)
        
        if num_iter%ncritic == 0:
            train_gen(args,noise,gen_lbls,critic_lbls)
            

            
def train(args):
    for epoch in range(*args['epochs']):

        if (epoch+1) % 10 == 0:
            print('saving generator')
            torch.save(args['gen']['model'].state_dict(),args['gen']['dir'])
            print('saving critic')
            torch.save(args['critic']['model'].state_dict(),args['critic']['dir'])
            
            print('saving generated images')
            img = generate_img(args)
            img_name = f'image_samples/generated_img_sample_epoch_{epoch+1}.pt'
            torch.save(img,img_name)
        
        print(f'Epoch: {epoch}')
        train_epoch(args)
        args['gen']['schedular'].step()
        args['critic']['schedular'].step()
        
        
args = {
    'critic':{
        'model': None,
        'ncritic':5,
        'optim':{
            'algorithm':None,
            'lr':0.0001,
            'betas':(0.5,0.99),
        },
        'schedular':None,
        'dir':'Critic.pth',
        'load':False,
        'lamda':10
    },
    'gen':{
        'model': None,
        'optim':{
            'algorithm':None,
            'lr':0.0001,
            'betas':(0.5,0.99)
        },
        'schedular':None,
        'dir':'Conditional_Convolutional_Generator.pth',
        'load':False
    },
    'data':{
        'loader': None,
        'batch_size':4,
        'img_shape':[1,28,28],
        'z_features':54,
        'constant_z':None,
        },
    'device':'cuda',
    'epochs':(0,3)
}

constant_z = torch.randn(10,64)
constant_z[:,-10:] = 0
for i in range(0,10):
    constant_z[i][i-10] = 1
args['data']['constant_z'] = constant_z.to(args['device'])

args['gen']['model'] = Generator().to(args['device'])
args['critic']['model'] = Critic().to(args['device'])

args['gen']['optim']['algorithm'] = optim.Adam(args['gen']['model'].parameters(),
                        lr=args['gen']['optim']['lr'],
                        betas=args['gen']['optim']['betas'])
args['critic']['optim']['algorithm'] = optim.Adam(args['critic']['model'].parameters(),
                        lr=args['critic']['optim']['lr'],
                        betas=args['critic']['optim']['betas'])

args['gen']['schedular'] = optim.lr_scheduler.ExponentialLR(args['gen']['optim']['algorithm'],gamma=0.9747)
args['critic']['schedular'] = optim.lr_scheduler.ExponentialLR(args['critic']['optim']['algorithm'],gamma=0.9747)

args['data']['loader'] = DataLoader(dataset=concatenated_dataset,
                        batch_size=args['data']['batch_size'],
                        shuffle=True,drop_last=True,
                        pin_memory=True,num_workers=2)
iter_loader = iter(args['data']['loader'])


train(args)