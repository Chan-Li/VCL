# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:40:42 2023

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

device = torch.device('cuda')

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

        w = torch.sign(self.h)
        return sign(x.mm(w.t()))
    def forward_sample(self, x):
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
        return (o==y).float().mean(), (o==y).float().sum()
    
    def accuracy_sample(self, x, y):
        o = self.forward_sample(x).reshape(-1)

        return (o==y).float().mean(), (o==y).float().sum()
    


class SGDoptim():
    def __init__(self, net, lr = 1e-3, gamma = 0.9, batch = 1, KL = True, device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.net = net
        self.lr = lr
        self.gamma = gamma
        self.v = torch.zeros(net.h.shape).to(device)
        self.KL = KL
        self.batch = batch
        

    def update(self, x, y):
        grad_exp = self.net.exp_grad(x * 1, y)
        if self.KL:
            grad_KL = self.net.KL_grad()
        else:
            grad_KL = 0
        grad = grad_exp.sum(0,keepdim = True) +grad_KL*self.batch
        self.v = self.gamma*self.v + self.lr*grad

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
        
        
        
def GDexp(alpha, bs, lr, weight_decay):
    N = 1000
    results = []
    resultss = []
    for times in range(20):
        print(alpha,' ', times, end = ';;')
        N_data = int(N*alpha)
    #mask_1 = torch.from_numpy(np.random.choice([0,1], size = [1, N] ,p=[1/2,1/2])).to(device).float()
        mask_1 = 1
        data = torch.from_numpy(np.random.choice([-1,1], size = [N_data, N])).to(device).float()*mask_1
        teacher = Perceptron()
        teacher.h = torch.from_numpy(np.random.choice([-1,1], size = [1, N])).to(device).float()

        label = teacher.forward(data).reshape(-1)


        t_data = torch.from_numpy(np.random.choice([-1,1], size = [10000, N])).to(device).float()*mask_1
        t_label = teacher.forward(t_data).reshape(-1)


        data_list = list(zip(data, label))

        re = []
        batch_size = bs
        dataloader = torch.utils.data.DataLoader(data_list, batch_size=batch_size, shuffle = False)


        net = Perceptron(beta = 1, N = N)
        net.h = torch.randn(1,N).to(device)
        opt = Adamoptim(net, lr = lr, KL = False, beta = [0.9,0.999], weight_decay = weight_decay)
        opt.batch = batch_size/N_data
        t1 = time.time()
        for j in range(2000):

            for (data, label) in dataloader:
                opt.update(data,label)
            if net.accuracy(t_data, t_label)[0] == 1:
                break
        print(time.time() - t1, bs, lr, weight_decay, torch.arccos((sign(net.h)*teacher.h).mean()).item()/np.pi )
        results.append([torch.arccos((sign(net.h)*teacher.h).mean()).item()/np.pi, (sign(net.h)*teacher.h).mean().item()])
    torch.save(results, 'data/GD_alpha='+str(alpha))





alpha_list = [i/10+0.1 for i in range(17)]
bs_list = [128,64,128,100,100,128,64,64,64,64,100,100,100,100,100,100,64]
lr_list = [0.008, 0.008, 0.002, 0.005, 0.008, 0.002, 0.002, 0.005, 0.002,
        0.002, 0.002, 0.005, 0.005, 0.005, 0.008, 0.008, 0.002]
weight_decay_list = [0.5 , 0.5 , 0.5 , 0.5 , 0.1 , 0.1 , 0.03, 0.03, 0.1 , 0.03, 0.1 ,
        0.1 , 0.1 , 0.1 , 0.05, 0.03, 0.1]

for i in range(17):
    GDexp(alpha= (i+1)/10, bs = bs_list[i], lr = lr_list[i], weight_decay= weight_decay_list[i])