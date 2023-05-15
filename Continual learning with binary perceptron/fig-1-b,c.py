# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:07:12 2023

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

        return (o==y).float().mean(), (o==y).float().sum()
    
    def accuracy_sample(self, x, y):
        o = self.forward_sample(x).reshape(-1)

        return (o==y).float().mean(), (o==y).float().sum()
    
    def accuracy_tri(self, x, y):
        o = sign( x.mm( double_step( torch.tanh(self.h) ).t())).reshape(-1)
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
        




def GDexp_second_task(alpha_1, alpha_2, r_0=0):
    results = []
    resultss = []
    for timesd in range(20):
        device = torch.device('cuda')
        beta = 1
        N = 1000
        N_data_1 = int(N*alpha_1)
        
        N_data_2 = int(N*alpha_2)

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
        

        re = []
        dataloader_1 = torch.utils.data.DataLoader(data_list_1, batch_size=32, shuffle = False)
        dataloader_2 = torch.utils.data.DataLoader(data_list_2, batch_size=32, shuffle = False)

        net = Perceptron(beta = beta, N = N)

        opt = SGDoptim(net, lr = 1e-3/beta, KL = False, gamma = 0.9)
        opt.batch = 32/N_data_2

        for j in range(3000):
            
            for (data, label) in dataloader_1:
                opt.update(data,label)
            if j %30==0:
                re.append([1- net.accuracy(t_data_1, t_label_1)[0].item(), (sign(net.h)*teacher.h).mean().item(), 
                           1- net.accuracy(t_data_2, t_label_2)[0].item(), (sign(net.h)*teacher2.h).mean().item(),(1-torch.tanh(net.h)**2).mean() ])
        opt = SGDoptim(net, lr = 1e-3/beta, KL = True, gamma = 0.9)
        opt.batch = 32/N_data_2
        net.h_last = net.h*1
        for j in range(3000):
            
            for (data, label) in dataloader_2:
                opt.update(data,label)
            if j %30==0:
                re.append([1- net.accuracy(t_data_1, t_label_1)[0].item(), (sign(net.h)*teacher.h).mean().item(), 
                           1- net.accuracy(t_data_2, t_label_2)[0].item(), (sign(net.h)*teacher2.h).mean().item(),(1-torch.tanh(net.h)**2).mean() ])
        results.append(re)
    torch.save(results, 'data/GD_alpha_second_exihibit='+str(alpha_1) +','+str(alpha_2)+'r_0='+str(r_0))



GDexp_second_task(2, 1, r_0=0)
GDexp_second_task(2, 2, r_0=0)
GDexp_second_task(2, 4, r_0=0)






alpha_list = [1,2,4]
r_0 = 0
alpha_1=2
l_mean = []
l_min = []
l_max = []
l_mean2 = []
l_min2 = []
l_max2 = []
sigma_mean = []
sigma_min = []
sigma_max = []
for alpha in alpha_list:
    results = torch.load('data/GD_alpha_second_exihibit='+str(alpha_1) +','+str(alpha)+'r_0='+str(r_0))
    results = torch.tensor(results)
    p1 = results[:,:,1]
    p2 = results[:,:,3]
    sigma = results[:,:,4]
    
    l1 = torch.arccos(p1)/np.pi
    l2 = torch.arccos(p2)/np.pi
    
    #print(sigma.mean(0))
    l_mean.append(l1.mean(0))
    l_min.append(l1.min(0)[0])
    l_max.append(l1.max(0)[0])
    l_mean2.append(l2.mean(0))
    l_min2.append(l2.min(0)[0])
    l_max2.append(l2.max(0)[0])
    sigma_mean.append(sigma.mean(0))
    sigma_min.append(sigma.min(0)[0])
    sigma_max.append(sigma.max(0)[0])   
    
    
plt.figure(figsize = (6,5))
axis = plt.subplot(111)
color = ['darkgray','dimgrey','black',]
for i in range(len(alpha_list)):
    plt.plot([j*30+1 for j in range(200)], l_mean[i],color = color[i], label = r'$\alpha_2=$'+str(alpha_list[i]))
    plt.fill_between([j*30 for j in range(200)], l_max[i].numpy(), l_min[i].numpy(),color = color[i], alpha = 0.3)
plt.tick_params(which = 'major', direction = 'in', width = 3, length = 6)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#plt.xlim(0,500)
plt.legend(fontsize = 15, loc = 'upper right')
plt.xlabel('epoch',fontsize = 30)
plt.ylabel(r'$\epsilon_g^1$', fontsize = 30)
axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1500))
plt.savefig('fig/fig-1-b.pdf', bbox_inches = 'tight')




plt.figure(figsize = (6,5))
axis = plt.subplot(111)
for i in range(len(alpha_list)):
    plt.plot([j*30+1 for j in range(200)], l_mean2[i] ,color = color[i], label = r'$\alpha_2=$'+str(alpha_list[i]))
    plt.fill_between([j*30 for j in range(200)], l_max2[i].numpy(), l_min2[i].numpy(),color = color[i], alpha = 0.3)
plt.tick_params(which = 'major', direction = 'in', width = 3, length = 6)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#plt.xlim(0,500)
plt.legend(fontsize = 15, loc = 'upper right')
plt.xlabel('epoch',fontsize = 30)
plt.ylabel(r'$\epsilon_g^2$', fontsize = 30)
axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1500))
plt.savefig('fig/fig-1-c.pdf', bbox_inches = 'tight')