import torch
from torch.autograd import grad as torch_grad

def one_hot(lbl):
    gen_lbl = torch.zeros(10)
    gen_lbl[lbl] = 1
    critic_lbl = torch.zeros(10,28,28)
    critic_lbl[lbl] = 1
    return gen_lbl,critic_lbl


def generate_img(args):
    args['gen']['model'].eval()
    with torch.no_grad():
        img = args['gen']['model'](args['data']['constant_z']).to('cpu')
    args['gen']['model'].train()
    return img

def wloss(critic_output,critic_output_gen):
    return torch.mean(critic_output_gen) - torch.mean(critic_output)

def reg(args,critic_input,critic_input_gen):
    
#     imgs_shape = (28,28,1)
    alpha = torch.rand(args['data']['batch_size'],1,1,1).to(args['device'])
    interpolated = alpha*critic_input.data + (1-alpha)*critic_input_gen.data
    interpolated.requires_grad = True
    pred_interpolated = args['critic']['model'](interpolated)
    
    gradient = torch_grad(outputs=pred_interpolated,
                          inputs=interpolated,
                          grad_outputs=torch.ones(args['data']['batch_size'],1).to(args['device']),
                          create_graph=True,
                          retain_graph=True)[0]
    
    gradient_norm = torch.sqrt(torch.sum(gradient**2,dim=(2,3))+1e-12)
    reg_value = torch.mean((gradient_norm-1)**2)
    return args['critic']['lamda']*reg_value