# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:03:03 2023

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
        #print(grad_exp.shape, grad_KL[0,0])
        #print(grad_exp.shape)
        grad = grad_exp.sum(0,keepdim = True) +grad_KL*self.batch
        self.m = self.beta[0]*self.m + (1-self.beta[0])*grad
        self.s = self.beta[1]*self.s + (1-self.beta[1])*grad**2
        self.m_hat = self.m/(1 - self.beta[0]**self.t)
        self.s_hat = self.s/(1-self.beta[1]**self.t)
        #print(grad_exp, grad_KL)
        #self.net.h_last = self.net.h*1
        self.net.h -= self.lr*(self.m_hat/(torch.sqrt(self.s_hat) + 1e-20) + self.weight_decay*self.net.h)
        #print(grad[0,0], self.net.h[0][0])
        self.t += 1
        
        
        
        
def GDexp(alpha):
    #To obtain the data of first task learning dynamic 
    results = []
    resultss = []
    for times in range(20):
        print(alpha,' ', times, end = ';;')
        device = torch.device('cuda')
        beta = 1
        N = 1000
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
        batch_size = max( int(np.ceil(N_data/20)), 10)
        dataloader = torch.utils.data.DataLoader(data_list, batch_size=batch_size, shuffle = False)


        net = Perceptron(beta = beta, N = N)

        opt = SGDoptim(net, lr = 3e-2/beta, KL = False, gamma = 0.9)
        opt.batch = 32/N_data

        for j in range(500):
            
            for (data, label) in dataloader:
                opt.update(data,label)
            
            
            re.append([1- net.accuracy(t_data, t_label)[0].item(), (sign(net.h)*teacher.h).mean().item(), (1-torch.tanh(net.h)**2).mean()])
        results.append(re.copy())
    torch.save(results, 'data/GD_alpha_exhibit='+str(alpha))



alpha_list = [0.5,1,1.5,2]
for i in alpha_list:
    GDexp(i)
    
l_mean = []
l_min = []
l_max = []
sigma_mean = []
sigma_min = []
sigma_max = []
for alpha in alpha_list:
    results = torch.load('data/GD_alpha_exhibit='+str(alpha))
    results = torch.tensor(results)
    p = results[:,:,1]
    sigma = results[:,:,2]
    
    l = torch.arccos(p)/np.pi
    #print(sigma.mean(0))
    l_mean.append(l.mean(0))
    l_min.append(l.min(0)[0])
    l_max.append(l.max(0)[0])
    sigma_mean.append(sigma.mean(0))
    sigma_min.append(sigma.min(0)[0])
    sigma_max.append(sigma.max(0)[0])  
    
    
# plot the data
plt.figure(figsize = (6,5))
axis = plt.subplot(111)
color = ['silver','darkgray','dimgray','black']
for i in range(len(alpha_list)):
    if i ==0:
        plt.plot([j+1 for j in range(500)], l_mean[i],color = color[i], label = r'$\alpha=$'+str(alpha_list[i]), linewidth = 3)
    else:
        plt.plot([j+1 for j in range(500)], l_mean[i],color = color[i], label = r'$\alpha=$'+str(alpha_list[i]), linewidth = 5)
    plt.fill_between([j+1 for j in range(500)], l_max[i].numpy(), l_min[i].numpy(), color = color[i],alpha = 0.3)
plt.tick_params(which = 'major', direction = 'in', width = 3, length = 6)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim(0,500)
plt.legend(fontsize = 15, loc = 'upper right')
#plt.axhline(0)
#axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
plt.xlabel('epoch',fontsize = 30)
plt.ylabel(r'$\epsilon_g$', fontsize = 30)
plt.savefig('fig/fig-1-a.pdf', bbox_inches = 'tight')