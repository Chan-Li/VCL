# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:23:17 2023

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

eps = 1e-50


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
        
## SGD data without fixed q_star       
        
device = torch.device('cuda')
q_star_list =[]
loss_list = []
for i in range(20):
    alpha = 2
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
    dataloader = torch.utils.data.DataLoader(data_list, batch_size=32, shuffle = False)


    net = Perceptron(beta = beta, N = N)

    opt = SGDoptim(net, lr = 1e-3/beta, KL = False, gamma = 0.9)
    opt.batch = 100/N_data
    q_star = []
    loss = []
    for j in range(4000):

        for (data, label) in dataloader:
            opt.update(data,label)
        if (j+1) % 100 == 0:
            print(1 - (torch.tanh(net.h)**2).mean(), net.accuracy(t_data, t_label)[0],(torch.sign( net.h)*teacher.h).mean(), ';;')
        q_star.append((torch.tanh(net.h)**2).mean().item()) 
        loss.append(1-net.accuracy(t_data, t_label)[0].item())
    q_star_list.append(q_star)
    loss_list.append(loss)
torch.save([q_star_list, loss_list], 'data/GD_dynamic_without_fixed_q_star')




##  SGD data with fixed q_star 

device = torch.device('cuda')
q_star_list =[]
loss_list = []
for i in range(20):
    alpha = 2
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
    dataloader = torch.utils.data.DataLoader(data_list, batch_size=32, shuffle = False)


    net = Perceptron(beta = beta, N = N)

    opt = SGDoptim(net, lr = 1e-3/beta, KL = False, gamma = 0.9)
    opt.batch = 100/N_data
    q_star = []
    loss = []
    for j in range(4000):

        for (data, label) in dataloader:
            opt.update(data,label)
            for i in range(100):
                net.h = torch.arctanh(torch.clamp(torch.tanh(net.h)/(torch.tanh(net.h)**2).mean()*(0.3 + 0.7/3000*j), -0.99999999, 0.99999999))
                if abs(torch.tanh(net.h)**2).mean()  - (0.3 + 0.7/3000*j)<1e-3:
                    break
        if (j+1) % 100 == 0:
            print(1 - (torch.tanh(net.h)**2).mean(), net.accuracy(t_data, t_label)[0],(torch.sign( net.h)*teacher.h).mean(), ';;')
        q_star.append((torch.tanh(net.h)**2).mean().item()) 
        loss.append(1-net.accuracy(t_data, t_label)[0].item())
    q_star_list.append(q_star)
    loss_list.append(loss)
torch.save([q_star_list, loss_list], 'data/GD_fix_q_star')




##   replica with fixed q_star


def partial_E(q, r, q_star, beta):
    z = torch.randn(10000,1).to(device).double()
    x = torch.randn(1,10000).to(device).double()
    g = (torch.sqrt(q_star - q + eps)*x  + torch.sqrt(q +eps)*z)/torch.sqrt(1-q_star +eps)
    
    D = ((H(g))**beta).mean(1, keepdim = True) + eps
    partial_r = H_prime( z*r/torch.sqrt(q-r**2 + eps) ) * (z/torch.sqrt(q-r**2 + eps) + z*r**2/(q-r**2 + eps)**(3/2) ) *torch.log(D) 
    partial_r = 2*partial_r.mean()
    
    
    partial_q_1 = H_prime( z*r/torch.sqrt(q-r**2 + eps) ) * ( - z*r/(q-r**2 + eps)**(3/2)/2 ) *torch.log(D) 
    
    partial_q_2 = H( z*r/torch.sqrt(q-r**2 + eps) ) * (beta * (H(g))**(beta-1) * H_prime(g) * \
                   1/2/torch.sqrt(1-q_star+eps)*(z/torch.sqrt(q + eps) - x/torch.sqrt(q_star - q + eps)) ).mean(1, keepdim = True)/D
   
    partial_q = 2*(partial_q_1.mean() + partial_q_2.mean())
    
    partial_q_star = H( z*r/torch.sqrt(q-r**2 + eps) ) * (beta * (H(g))**(beta-1) * H_prime(g) * \
                    ( (x*torch.sqrt(q_star - q + eps) + torch.sqrt(q)*z)/2/(1-q_star)**(3/2) + \
                    x/2/torch.sqrt(1-q_star+eps)/torch.sqrt(q_star - q + eps)  ) ).mean(1, keepdim = True)/D

    partial_q_star = 2*partial_q_star.mean()
    
    return partial_q, partial_r, partial_q_star



def partial_S(q,r,q_star, beta):
    z = torch.randn(5000,1).to(device).double()
    m = 2*torch.rand(1,5000).to(device).double() -1
    h = (q_star - q/2 + eps)*m**2 + torch.sqrt(q + eps)*z*m + r*m
    D = torch.exp(h).mean(1,keepdim = True) + eps
    M = torch.exp(h)
    partial_q = ((M*(-m**2/2 + m*z/2/(torch.sqrt(q + eps)) )).mean(1,keepdim = True) \
                 /D).mean()
    partial_r = ( (M*m).mean(1,keepdim=True)/D).mean()
    partial_q_star = ( (M*m**2).mean(1,keepdim=True)/D).mean()
    return partial_q, partial_r, partial_q_star



def partial_S_fixed_q_star(q,r,fixed_q_star, beta, init = 0):
    z = torch.randn(5000,1).to(device).double()
    m = 2*torch.rand(1,5000).to(device).double() -1
    
    for i in range(10):
        q_star = torch.rand(1).double().to(device) + init
        for k in range(200):
            h = (q_star - q/2 + eps)*m**2 + torch.sqrt(q + eps)*z*m + r*m

            D = torch.exp(h).mean(1,keepdim = True) + eps
            M = torch.exp(h)
            f = ( (M*m**2).mean(1,keepdim=True)/D).mean() - fixed_q_star
            f_prime = ( (M*m**4).mean(1,keepdim=True)/D).mean() - \
                    ( (M*m**2).mean(1,keepdim=True)**2/D**2).mean()


            q_star = q_star - f/f_prime
            if abs(f/f_prime)<1e-5:
                break
            if q_star != q_star:
                break
        if abs(f/f_prime)<1e-5:
            return q_star
    print('n converge', abs(f/f_prime) )
    return torch.tensor(torch.nan).to(device)





def fix_q_star(alpha, beta, q_star_):
    q_star = torch.tensor(q_star_).to(device)
    beta_list = [beta]
    alpha = alpha
    results = []
    results_hat = []
    resultss = []
    eta = 0.9
    q_hat = r_hat =q_star_hat = 0
    for x in range(20):
        for l in range(len(beta_list)):
            beta = beta_list[l]
            r = torch.rand(1).to(device).double()
            q =  torch.rand(1).to(device).double()*r

            for i in range(100):
                temp = [q*1 , r*1, q_star*1]

                temp1 = torch.zeros([20,3]).to(device).double()
                j = 0
                k = 0
                n_fails = 0
                while(j<20):
                    temp1[j,0],temp1[j,1],temp1[j,2] = partial_E(q, r, q_star, beta)
                    if not (temp1[j]!=temp1[j]).sum():
                        j+=1
                        n_fails = 0
                    else:
                        n_fails += 1

                    if n_fails > 5:
                        r = torch.rand(1).to(device).double()
                        q =  torch.rand(1).to(device).double()*r
                        
                        #q_star = torch.rand(1).to(device).double()*(1-q) +q
                        n_fails = 0
                        print('change init', end = 'emmm')

                    k += 1

                    if k>100:
                        print('not complete!',j,end='=e')
                        break


                q_hat_temp, r_hat_temp, q_star_hat_temp =  temp1[:j,:].mean(0)
                q_hat = q_hat + eta*((-2*alpha)*q_hat_temp - q_hat)
                r_hat = r_hat + eta*(alpha*r_hat_temp - r_hat)
                q_star_hat = partial_S_fixed_q_star(q_hat*1,r_hat*1,q_star*1,  beta, init = q_star_hat_temp ,)

                j = 0
                k = 0
                temp1 = torch.zeros([20,3]).to(device).double()
                while(j<20):
                    temp1[j][0],temp1[j][1],temp1[j][2]  = partial_S(q_hat, r_hat, q_star_hat, beta)
                    if not (temp1[j]!=temp1[j]).sum():
                        j+=1     
                    k += 1
                    if k>100:
                        print('not complete!',j,end='=j')
                        break
                q_temp,r_temp,_ = temp1[:j,:].mean(0)
                q_temp *= -2
                q = q + (eta)*(q_temp -q)
                r = r+ (eta)*(r_temp-r)

                if q <r**2:
                    r = torch.sqrt(q) - (torch.sqrt(q)- q)*torch.rand(1).to(device).double()
                #print(q,r,q_star)
                err = ((temp[0]-q)**2 + (temp[1]-r)**2 ).mean().item()
                #print('....',temp, q, r, q_star ,  end = ';' )
                #print(err)
                if (i == 99) or ( err <1e-5):
                    results.append([q.item(), r.item(), q_star.item()])
                    results_hat.append([q_hat.item(), r_hat.item(), q_star_hat.item()])
                    #print('alpha',alpha,'beta',beta, ';',i,';',err)
                    #print(q,r,q_star,q_hat, r_hat, q_star_hat, end = ';;;')
                    #torch.save([results,results_hat], 'results_alpha=0.8')
                    break
        
        resultss.append([results, results_hat])
    r_temp = torch.tensor(results_hat).cuda()
    l_list = []
    p_list = []
    for i in range(5):
        print(i)
        l,p = loss(r_temp[:,0].mean(), r_temp[:,1].mean(), r_temp[:,2].mean())
        l_list.append(l)
        p_list.append(p)
    torch.save([results,results_hat, l_list, p_list], 'data/results_grid_fix_q_star'+str(q_star_))
    print('error',l_list, p_list)



q_star_list = [0.75 +i/100 for i in range(25)]
for i in q_star_list:
    fix_q_star(2, 5, i)
    
    
    
    



##   plot the data

results =[]
results_hat = []
l_list = []
p_list = []
for i in range(25):
    print(i)
    a,b,c,d = torch.load('data/results_grid_fix_q_star'+str(q_star_list[i]))
    results.append(np.array(a).mean(0))
    results_hat.append(np.array(b).mean(0))


results = np.array(results)
results_hat = np.array(results_hat)
p_list = torch.tensor(p_list).cpu().numpy()
l_list = torch.tensor(l_list).cpu().numpy()
device = torch.device('cuda')
eps = 1e-30
l_list = []
p_list = []
for j in range(30):
    l, p =[], []
    for i in range(25):
        temp = torch.from_numpy(results_hat[i]).cuda()
        a,b = loss(temp[0], temp[1], temp[2])
        l.append(a.item())
        p.append(b.item())
    l_list.append(l)
    p_list.append(p)
l = np.array(l_list)
p = np.array(p_list)





plt.figure(figsize = (6,5))
axis = plt.subplot(111)
q_star_GD, l_GD = torch.load('data/GD_dynamic_without_fixed_q_star')
q_star_GD, l_GD = np.array(q_star_GD), np.array(l_GD)
q_star_fix, l_fix = torch.load('data/GD_fix_q_star')
q_star_fix, l_fix = np.array(q_star_fix), np.array(l_fix)
q_star_fix = np.array(q_star_fix)
l_fix = np.array(l_fix)
q_star_GD = np.array(q_star_GD)
l_GD = np.array(l_GD)
#plt.errorbar(results[:,2][results[:,2]>0.7], l.mean(0)[results[:,2]>0.7], l.std(0)[results[:,2]>0.7], label = 'Anneal', linewidth =5)

plt.errorbar(q_star_GD[:,q_star_GD.mean(0)>=0.75].mean(0)[::50], l_GD[:,q_star_GD.mean(0)>=0.75].mean(0)[::50], 
             xerr = q_star_GD[:,q_star_GD.mean(0)>=0.75].std(0)[::50], yerr = l_GD[:,q_star_GD.mean(0)>=0.75].std(0)[::50], 
             capsize = 3,markerfacecolor="white",marker = 'o',markersize = 10,label = 'SGD', linewidth =2)
plt.plot([0.74+0.2*(i+1)/20 for i in range(25)], l.mean(0),'s-',markersize = 7,markerfacecolor="white",label = 'replica', linewidth =5)
plt.errorbar(q_star_fix[:,(q_star_fix).mean(0)>=0.75].mean(0)[::50], l_fix[:, (q_star_fix).mean(0)>=0.75].mean(0)[::50],
             xerr = q_star_fix[:,(q_star_fix).mean(0)>=0.75].std(0)[::50],yerr = l_fix[:, (q_star_fix).mean(0)>=0.75].std(0)[::50], 
             capsize = 3,markerfacecolor="white",marker = '^',markersize = 10,label = 'SGD slow', linewidth =2)

plt.xlabel(r'$q_\star$', fontsize = 30)
plt.ylabel(r'$\epsilon_g$', fontsize = 30)
plt.legend(fontsize = 15)
#plt.xlabel(r'$\alpha$', fontsize = 20)
plt.tick_params(which = 'major', direction = 'in', width = 4, length = 8)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
plt.savefig('fig/fig-2-d.pdf', bbox_inches = 'tight')







