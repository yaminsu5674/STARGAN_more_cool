from model import Generator
from model import Discriminator

import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable



class Solver(object):

    def __init__(self, celeba_loader, config):
        self.celeba_loader= celeba_loader

        #model_configurations
        self.c_dim= config.c_dim
        self.c2_dim= config.c2_dim
        self.g_conv_dim= config.g_conv_dim
        self.d_conv_dim= config.d_conv_dim
        self.g_repeat_num= config.g_repeat_num
        self.d_repeat_num= config.d_repeat_num
        self.lambda_cls= config.lambda_cls
        self.lambda_rec= config.lambda_rec
        self.lambda_gp= config.lambda_gp

        #Training configurations
        self.num_iters= config.num_iters
        self.num_iters_decay= config.num_iters_decay
        self.g_lr= config.g_lr
        self.d_lr= config.d_lr
        self.n_critic= config.n_critic
        self.beta1= config.beta1
        self.beta2= config.beta2
        self.model_save_step= config.model_save_step
        self.lr_update_step= config.lr_update_step

        #Miscellaneous
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir= config.model_save_dir
        
        self.build_model()

    
    def build_model(self):
        self.G= Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D= Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer= torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer= torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.G.to(self.device)
        self.D.to(self.device)



    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr']= g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr']= d_lr



    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()



    def gradient_penalty(self, y, x):
        weight= torch.ones(y.size()).to(self.device)
        dydx= torch.autograd.grad(outputs= y,
        inputs= x,
        grad_outputs= weight,
        retain_graph= True,
        create_graph= True,
        only_inputs= True)[0]

        dydx= dydx.view(dydx.size(0), -1)
        dydx_l2norm= torch.sqrt(torch.sum(dydx**2, dim= 1))
        return torch.mean((dydx_l2norm-1)**2)

    def classification_loss(self, logit, target):
        return F.binary_cross_entropy_with_logits(logit, target, size_average= False)/ logit.size(0)



    def train(self):
        data_loader= self.celeba_loader

        data_iter= iter(data_loader)
        training_epochs= self.num_iters

        g_lr= self.g_lr
        d_lr= self.d_lr

        for i in range(training_epochs):
            ori_img, ori_label= next(data_iter)

            rand_idx= torch.randperm(ori_label.size(0))
            trg_label= ori_label[rand_idx]

            ori_img= ori_img.to(self.device)
            ori_label= ori_label.to(self.device)
            trg_label= trg_label.to(self.device)

            # -----------------Discrdiminator Training-------------------

            out_src, out_cls= self.D(ori_img)
            d_loss_real= -torch.mean(out_src)
            d_loss_cls= self.classification_loss(out_cls, ori_label)

            x_fake= self.G(ori_img, trg_label)
            out_src, out_cls= self.D(x_fake.detach())
            d_loss_fake= torch.mean(out_src)

            alpha= torch.rand(ori_img.size(0), 1, 1, 1).to(self.device)
            x_hat= (alpha * ori_img.data + (1-alpha) * x_fake.data).requires_grad_(True)
            out_src, _= self.D(x_hat)
            d_loss_gp= self.gradient_penalty(out_src, x_hat)


            d_loss= d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # -----------------Discrdiminator Training-------------------

            if (i + 1) % self.n_critic == 0:
                x_fake= self.G(ori_img, trg_label)
                out_src, out_cls= self.D(x_fake)
                g_loss_fake= -torch.mean(out_src)
                g_loss_cls= self.classification_loss(out_cls, trg_label)

                x_reconst= self.G(x_fake, ori_label)
                g_loss_rec= -torch.mean(torch.abs(ori_img - x_reconst))

                g_loss= g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

            if (i + 1) % 200 == 0 and (i + 1) != 1:
                print('{}th epochs,  G_loss : {},  D_loss : {}'.format(i, g_loss, d_loss))


            if (i + 1)% self.lr_update_step == 0 and (i + 1) > (training_epochs - 100000):
                g_lr-= g_lr / float(100000)
                d_lr-= d_lr / float(100000)
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr : {}, d_lr : {}.'.format(g_lr, d_lr))

            
            if (i + 1) % self.model_save_step == 0:
                G_path= os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path= os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Save model checkpoints !')











    
