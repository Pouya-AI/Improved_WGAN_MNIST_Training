import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models.Generator import Generator
from models.Critic import Critic
from utils import wloss,reg,generate_img
from tqdm import tqdm


transform = T.ToTensor()
# dataset_train = MNIST(root='mnist_train',train=True,transform=transform,download=True)
dataset_test = MNIST(root='mnist_test',train=False,transform=transform,download=False)
# concatenated_dataset = ConcatDataset([dataset_train,dataset_test])
concatenated_dataset = dataset_test


def train_critic(args, imgs_real, noises):
    args['critic']['model'].train()
    args['gen']['model'].eval()
    with torch.no_grad():
        imgs_gen = args['gen']['model'](noises)
    preds_real = args['critic']['model'](imgs_real)
    preds_gen = args['critic']['model'](imgs_gen)
    wloss_value = wloss(preds_real, preds_gen)
    reg_value = reg(args, args['critic']['model'], imgs_real, imgs_gen)
    loss_value = wloss_value + reg_value
    loss_value.backward()
    args['critic']['optim']['algorithm'].step()
    args['critic']['optim']['algorithm'].zero_grad()
    return loss_value.item(), wloss_value.item(), reg_value


def train_gen(args, noises):
    args['critic']['model'].eval()
    args['gen']['model'].train()
    imgs_gen = args['gen']['model'](noises)
    preds_gen = args['critic']['model'](imgs_gen)
    loss_value = -1 * torch.mean(preds_gen)
    loss_value.backward()
    args['gen']['optim']['algorithm'].step()
    args['gen']['optim']['algorithm'].zero_grad()
    args['critic']['model'].zero_grad()
    return loss_value.item()


def train_epoch(args):
    num_iter = 0
    for imgs_real, _ in tqdm(loader):
        num_iter += 1
        imgs_real = imgs_real.to(args['device'])
        noises = torch.randn(args['data']['batch_size'], args['data']['z_features']).to(args['device'])
        loss_value, wloss_value, reg_value = train_critic(args, imgs_real, noises)
        if num_iter % args['critic']['ncritic'] == 0:
            loss_value = train_gen(args, noises)


def train(args):
    for epoch in range(*args['epochs']):
        if (epoch + 1) % 1 == 0:
            print('saving generator')
            torch.save(args['gen']['model'].state_dict(), args['gen']['dir'])
            print('saving critic')
            torch.save(args['critic']['model'].state_dict(), args['critic']['dir'])
            # print('saving generated images')
            # img = generate_img(args)
            # img_name = f'image_samples/generated_img_sample_epoch_{epoch}.pt'
            # torch.save(img, img_name)
        if (epoch + 1) % 1 == 0:
            print('\n schedular step')
            args['critic']['schedular'].step()
            args['gen']['schedular'].step()
        print(f"Epoch: {epoch + 1}")
        train_epoch(args)



args = {
    'critic':{
        'model': None,
        'ncritic':3,
        'optim':{
            'algorithm':None,
            'lr':0.0001,
            'betas':(0.5,0.99),
        },
        'schedular':None,
        'dir':'critic.pth',
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
        'dir':'Unconditional_Convolutional_Generator.pth',
        'load':False
    },
    'data':{
        'loader': None,
        'batch_size':512,
        'img_shape':[1,28,28],
        'z_features':32,
        'constant_z':torch.randn(16,32),
        },
    'device':'cuda',
    'epochs':(0,200)
}


gen = Generator().to(args['device'])
args['gen']['model'] = gen
if args['gen']['load'] == True:
    print("loading Generator's state dictionary")
    args['gen']['model'].load_state_dict(torch.load(args['gen']['dir']))
critic = Critic().to(args['device'])
args['critic']['model'] = critic
if args['critic']['load'] == True:
    print("loading Critic's state dictionary")
    args['critic']['model'].load_state_dict(torch.load(args['critic']['dir']))

optim_gen = optim.Adam(gen.parameters(),lr=args['gen']['optim']['lr'],
                       betas=args['gen']['optim']['betas'])
args['gen']['optim']['algorithm'] = optim_gen

schedular_gen = optim.lr_scheduler.ExponentialLR(args['gen']['optim']['algorithm'],gamma=0.7)
args['gen']['schedular'] = schedular_gen

optim_critic = optim.Adam(critic.parameters(),lr=args['critic']['optim']['lr'],
                          betas=args['critic']['optim']['betas'])
args['critic']['optim']['algorithm'] = optim_critic

schedular_scritic = optim.lr_scheduler.ExponentialLR(args['critic']['optim']['algorithm'],gamma=0.7)
args['critic']['schedular'] = schedular_scritic

loader = DataLoader(dataset=concatenated_dataset,batch_size=args['data']['batch_size'],
                    shuffle=True,drop_last=True,pin_memory=True,num_workers=2)
args['data']['loader'] = loader

iter_loader = iter(loader)





train(args)