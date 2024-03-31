import torch
from torch.autograd import grad as torch_grad

def wloss(preds_real,preds_gen):
    return torch.mean(preds_gen) - torch.mean(preds_real)

def reg(args,critic,imgs_real,imgs_gen):
#     imgs_shape = (28,28,1)
    alpha = torch.rand(args['data']['batch_size'],1,1,1).to(args['device'])
    interpolated = alpha*imgs_real.data + (1-alpha)*imgs_gen.data
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


def generate_img(args):
    z = args['data']['constant_z'].to(args['device'])
    args['gen']['model'].eval()
    with torch.no_grad():
        img = args['gen']['model'](z).to('cpu')
    return img


