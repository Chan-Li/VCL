# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:38:29 2023

@author: huang
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl





def H(x):
    y = -1/2*torch.erf(x/np.sqrt(2)) + 1/2
    return y

def H_prime(x):
    return -torch.exp(-x*x/2)/np.sqrt(2*np.pi)

def sign(y):
    x = y*1
    x[x<0] = -1
    x[x>=0] = 1
    return x

def double_step(x):
    a = 0.5
    x[abs(x)<a] = 0
    x[abs(x)>=a] = torch.sign(x[abs(x)>=a])
    return x

class Perceptron():
    def __init__(self, N = 784, beta = 1, device = torch.device('cuda')):
        self.beta = beta
        self.h = torch.randn(1,N).to(device)
        self.h_last = self.h*1
        self.w = sign(self.h)
        self.sigma = 1-torch.tanh(self.beta*self.h)**2
        self.device = device
        self.N = N
    def forward(self,x):
        #print(x.shape, self.h)
        #w = sign(  1/(1+torch.exp(-2*self.beta*self.h)) - torch.rand(self.h.shape).to(self.device))
        w = torch.sign(self.h)
        return sign(x.mm(w.t()))
    def forward_sample(self, x):
        #w = torch.tanh(self.h)
        w = torch.exp(self.h)/2/torch.cosh(self.h) - torch.rand(1, self.N).cuda()
        return sign(x.mm(w.t()))
    
    
    def exp_grad(self,x,y):
        self.sigma_update()
        sigma_sum = self.sigma.sum()
        
        temp = - (y.reshape(-1,1)* (torch.tanh(self.beta*self.h)*x).sum(1,keepdim = True)  )/(torch.sqrt(sigma_sum)+1e-10)

        return  ( y.reshape(-1,1)*self.beta*self.sigma*(x*sigma_sum + torch.tanh(self.beta*self.h)* \
                                    (torch.tanh(self.beta*self.h)*x).sum(1,keepdim = True))/(sigma_sum**(3/2)+1e-10) \
                *H_prime(temp)\
                /(H(temp) +1e-10))
    
    def energy(self, x, y):
        sigma_sum = self.sigma.sum()
        temp = - (y.reshape(-1,1)* (torch.tanh(self.beta*self.h)*x).sum(1,keepdim = True)  )/(torch.sqrt(sigma_sum)+1e-10)
        
        return torch.log(H(temp) + 1e-15).mean()
    
        
    def KL_grad(self):
        self.sigma_update()
        return self.beta**2*(self.h - self.h_last)*self.sigma
    
    def sigma_update(self):
        self.sigma =  1-torch.tanh(self.beta*self.h)**2
        
    def accuracy(self, x, y):
        o = self.forward(x).reshape(-1)
        #print(o, y, o==y)
        return (o==y).float().mean(), (o==y).float().sum()
    
    def accuracy_sample(self, x, y):
        o = self.forward_sample(x).reshape(-1)
        #print(o, y, o==y)
        return (o==y).float().mean(), (o==y).float().sum()
    
    def accuracy_tri(self, x, y):
        o = sign( x.mm( double_step( torch.tanh(self.h) ).t())).reshape(-1)
        #print(o, y, o==y)
        return (o==y).float().mean(), (o==y).float().sum()


class SGDoptim():
    def __init__(self, net, lr = 1e-3, gamma = 0.9, batch = 1, weight_decay = 0, KL = True, device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.net = net
        self.lr = lr
        self.gamma = gamma
        self.v = torch.zeros(net.h.shape).to(device)
        self.KL = KL
        self.batch = batch
        self.weight_decay = weight_decay
        

    def update(self, x, y):
        grad_exp = self.net.exp_grad(x * 1, y)
        if self.KL:
            grad_KL = self.net.KL_grad()
        else:
            grad_KL = 0
        #print(grad_exp.shape, grad_KL[0,0])
        #print(grad_exp.shape)
        grad = grad_exp.sum(0,keepdim = True) +grad_KL*self.batch
        #print(grad_exp, grad_KL)
        #self.net.h_last = self.net.h*1
        self.v = self.gamma*self.v + self.lr*grad
        #print(grad[0,0], self.net.h[0][0])
        self.net.h -= self.v
      

        
        
class Adamoptim():
    def __init__(self, net, lr = 1e-3, beta = [0.9, 0.999], batch = 1, KL = True, weight_decay = 0,
                 device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.net = net
        self.lr = lr
        self.v = torch.zeros(net.h.shape).to(device)
        self.KL = KL
        self.batch = batch
        self.t = 1
        self.m = 0
        self.m_hat = 0
        self.s = 0
        self.s_hat = 0
        self.beta = beta
        self.weight_decay = weight_decay

    def update(self, x, y):
        
        grad_exp = self.net.exp_grad(x * 1, y)
        if self.KL:
            grad_KL = self.net.KL_grad()
        else:
            grad_KL = 0

        grad = grad_exp.sum(0,keepdim = True) +grad_KL*self.batch
        self.m = self.beta[0]*self.m + (1-self.beta[0])*grad
        self.s = self.beta[1]*self.s + (1-self.beta[1])*grad**2
        self.m_hat = self.m/(1 - self.beta[0]**self.t)
        self.s_hat = self.s/(1-self.beta[1]**self.t)
        self.net.h -= self.lr*(self.m_hat/(torch.sqrt(self.s_hat) + 1e-20) + self.weight_decay*self.net.h)

        self.t += 1
        


def GDexp_second_task(alpha_1, alpha_2, lr, bs, weight_decay, r_0=0):
    results = []
    resultss = []
    N=1000
    t1 = time.time()
    data_list = []
    for k in range(20):
        N = 1000
        N_data_1 = int(N*alpha_1)

        N_data_2 = int(N*alpha_2)

        #mask_1 = torch.from_numpy(np.random.choice([0,1], size = [1, N] ,p=[1/2,1/2])).to(device).float()
        mask_1 = 1
        data_1 = torch.from_numpy(np.random.choice([-1,1], size = [N_data_1, N])).to(device).float()*mask_1
        teacher = Perceptron()
        teacher.h = torch.from_numpy(np.random.choice([-1,1], size = [1, N])).to(device).float()

        label_1 = teacher.forward(data_1).reshape(-1)

        t_data_1 = torch.from_numpy(np.random.choice([-1,1], size = [10000, N])).to(device).float()*mask_1
        t_label_1= teacher.forward(t_data_1).reshape(-1)

        mask_1 = 1
        data_2 = torch.from_numpy(np.random.choice([-1,1], size = [N_data_2, N])).to(device).float()*mask_1
        teacher2 = Perceptron()
        teacher2.h = torch.from_numpy(np.random.choice([-1,1], p = [1/2-r_0/2, 1/2 + r_0/2], size = [1, N])).to(device).float()*teacher.h

        label_2 = teacher2.forward(data_2).reshape(-1)

        t_data_2 = torch.from_numpy(np.random.choice([-1,1], size = [10000, N])).to(device).float()*mask_1
        t_label_2 = teacher2.forward(t_data_2).reshape(-1)


        data_list_1 = list(zip(data_1, label_1))
        data_list_2 = list(zip(data_2, label_2))

        #batch_size = min([bs, N_data_2])
        re = []
        #bs = 16
        #lr = 5e-3
        #weight_decay = 1e-3
        dataloader_1 = torch.utils.data.DataLoader(data_list_1, batch_size = 64, shuffle = False)
        dataloader_2 = torch.utils.data.DataLoader(data_list_2, batch_size = bs, shuffle = False)

        net = Perceptron(beta = 1, N = N,device = device)

        #opt = Adamoptim(net, lr = 1e-3, KL = False, beta = [0.9,0.999], weight_decay = 1e-2)
        opt = SGDoptim(net, lr = 2e-4, KL = False, device = device, gamma = 0.9)
        #opt.batch = 32/N_data_1


        for j in range(4500):
            for (data, label) in dataloader_1:
                opt.update(data,label)
            #if (j+1) % 100 == 0:
                '''if opt.lr >1e-3:
                    opt.lr*=0.95'''
                #print(j,torch.arccos(((torch.sign(net.h))* torch.sign(teacher.h)).float().mean())/np.pi, torch.arccos(((torch.sign(net.h)) *torch.sign(teacher2.h)).float().mean())/np.pi, ';;')
            if ((torch.sign(net.h)) *torch.sign(teacher.h)).float().mean() == 1:
                break
              #  print(1 - (torch.tanh(net.h)**2).mean(),net.accuracy_tri(t_data, t_label)[0], (double_step(torch.tanh(net.h)) == torch.sign(teacher.h)).float().mean(),  net.accuracy_tri(t_data_2, t_label_2)[0], (double_step(torch.tanh(net.h)) == torch.sign(teacher2.h)).float().mean(), ';;')
        #print(torch.arccos(((torch.sign(net.h))* torch.sign(teacher.h)).float().mean())/np.pi, torch.arccos(((torch.sign(net.h)) *torch.sign(teacher2.h)).float().mean())/np.pi)
        opt = SGDoptim(net, lr = lr, KL = True, device = device, gamma = 0.95)
        opt.batch = bs/N_data_2
        net.h_last = net.h*1
        print(torch.arccos(((torch.sign(net.h))* torch.sign(teacher.h)).float().mean())/np.pi, torch.arccos(((torch.sign(net.h)) *torch.sign(teacher2.h)).float().mean())/np.pi, ';;')
        for j in range(7500):
            for (data, label) in dataloader_2:
                opt.update(data,label)
            #if (j+1) % 100 == 0:
             #   print(torch.arccos(((torch.sign(net.h))* torch.sign(teacher.h)).float().mean())/np.pi, torch.arccos(((torch.sign(net.h)) *torch.sign(teacher2.h)).float().mean())/np.pi, ';;')
            if ((torch.sign(net.h)) *torch.sign(teacher2.h)).float().mean() == 1:
                break
        data_list.append([torch.arccos(((torch.sign(net.h))* torch.sign(teacher.h)).float().mean())/np.pi, torch.arccos(((torch.sign(net.h)) *torch.sign(teacher2.h)).float().mean())/np.pi])

    #data_record.append([r_0, alpha_2, lr, bs,weight_decay, data_list.copy()])
    print(data_list, time.time()-t1, r_0, lr, bs, weight_decay)
    torch.save(data_list, 'data/GD_alpha_second='+str(alpha_1) +','+str(alpha_2)+'r_0='+str(0)+'-alpha1=1.4')


device = torch.device('cpu')
bs_list = [ 50, 100, 100,  50,  50,  50,  50, 100,  50]
lr_list = [0.0005, 0.0005, 0.001 , 0.005 , 0.001 , 0.001 , 0.001 , 0.0005,
        0.0005]
alpha_2_list = [(i+1)/2 for i in range(9)]



for i in range(len(alpha_2_list)):
    GDexp_second_task(alpha_1=1.4, alpha_2 = alpha_2_list[i],
                          r_0 = 0, lr = lr_list[i], 
                          bs = bs_list[i] , weight_decay = 1e-3)