# -*- coding: utf-8 -*-
"""
Created on Sat May 13 21:40:01 2023

@author: huang
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from scipy import integrate

device = torch.device('cuda')
eps = torch.exp( torch.ones(1).double().cuda() )**(-700)

def H(x):
    y = -1/2*torch.erf(x/np.sqrt(2)) + 1/2
    return y

def H_prime(x):
    return -torch.exp(-x**2/2)/np.sqrt(2*np.pi)


def partial_E(q, r, q_star, beta):
    z = torch.randn(10000,1).to(device).double()
    x = torch.randn(1,20000).to(device).double()
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
    x =z =g=D =0
    
    return partial_q, partial_r, partial_q_star




def partial_S(q,r,q_star, beta):
    k = beta
    z = torch.randn(10000,1).to(device).double()
    m = 2*torch.rand(1,20000).to(device).double() -1
    f = -(1+m)/2*torch.log((1+m)/2 + eps)  -(1-m)/2*torch.log((1-m)/2 +eps)
    h = (q_star - q/2 + eps)*m**2 + torch.sqrt(q + eps)*z*m + r*m + k*f
    h = h - h.max()
    D = torch.exp(h).mean(1,keepdim = True) + eps
    M = torch.exp(h)

    
    intterm = M/D
    intterm[D.reshape(-1)==0,:] = 0
    partial_q = ((-m**2/2 + m*z/2/(torch.sqrt(q + eps)))*intterm).mean()
    partial_r = (( m  )*intterm).mean()
    partial_q_star = (( m**2  )*intterm).mean()
    
    z = m=f=h=D=M=0

    return partial_q, partial_r, partial_q_star

def beta_exp(alpha, beta):
    beta_list = [beta]
    alpha = alpha
    results = []
    results_hat = []
    resultss = []
    for x in range(5):
        for l in range(len(beta_list)):
            beta = beta_list[l]
            
            q =  torch.rand(1).to(device).double()
            r = torch.rand(1).to(device).double()*q
            q_star = torch.rand(1).to(device).double()*(1-q) +q
            '''
            q, r,q_star = torch.ones(3).cuda().double()
            q *= 0.9232
            r *= 0.9575
            q_star *= 0.9237'''
            q_hat = 0
            r_hat = 0
            q_star_hat = 0
            eta = 0.9
            for i in range(200):
                if eta >0.9:
                    eta -= 0.05
                temp = [q*1 , r*1, q_star*1]
                if alpha >=2.2:
                    avrn = 40
                elif alpha>2:
                    avrn = 20
                else:
                    avrn = 20
                temp1 = torch.zeros([avrn,3]).to(device).double()
                j = 0
                k = 0
                n_fails = 0
                
                while(j<avrn):
                    temp1[j,0],temp1[j,1],temp1[j,2] = partial_E(q, r, q_star, beta)
                    if not (temp1[j]!=temp1[j]).sum():
                        j+=1
                        n_fails = 0
                    else:
                        n_fails += 1

                    if n_fails > 5:
                        q =  torch.rand(1).to(device).double()
                        r = torch.rand(1).to(device).double()*q
                        q_star = torch.rand(1).to(device).double()*(1-q) +q
                        n_fails = 0
                        print('change init', end = 'emmm')

                    k += 1

                    if k>100:
                        print('not complete!',j,end='=e')
                        break


                q_hat_temp, r_hat_temp, q_star_hat_temp =  temp1[:j,:].mean(0)
                q_hat_temp *= (-2*alpha)
                r_hat_temp *= alpha
                q_star_hat_temp *= alpha
                
                q_hat = q_hat + eta*(q_hat_temp-q_hat)
                r_hat = r_hat + eta*(r_hat_temp-r_hat)
                q_star_hat = q_star_hat + eta*(q_star_hat_temp-q_star_hat)
                print('.', q, r, q_star , q_hat,r_hat,q_star_hat, end = ';' )
                j = 0
                k = 0
                temp1 = torch.zeros([avrn,3]).to(device).double()
                
                while(j<avrn):
                    temp1[j][0],temp1[j][1],temp1[j][2]  = partial_S(q_hat, r_hat, q_star_hat, beta)
                    if not (temp1[j]!=temp1[j]).sum():
                        j+=1     
                    k += 1
                    if k>100:
                        print('not complete!',j,end='=j')
                        break
                q_temp,r_temp,q_star_temp = temp1[:j,:].mean(0)
                q_temp *= -2
                
                q = q + eta*(q_temp-q)
                r = r + eta*(r_temp-r)
                q_star = q_star + eta*(q_star_temp - q_star)
                
                #print(q,r,q_star)
                err = ((temp[0]-q)**2 + (temp[1]-r)**2 + (temp[2]-q_star)**2).mean().item()
                print(err)
                if (i == 199) or ( err <1e-5*eta**2):
                    results.append([q.item(), r.item(), q_star.item()])
                    results_hat.append([q_hat.item(), r_hat.item(), q_star_hat.item()])
                    print('alpha',alpha,'beta',beta, ';',i,';',err)
                    print(q,r,q_star,q_hat, r_hat, q_star_hat, end = ';;;')
                    #torch.save([results,results_hat], 'results_alpha=0.8')
                    break
        
        resultss.append([results, results_hat])
    r_temp = torch.tensor(results_hat).cuda()
    l_list = []
    p_list = []
#     for i in range(5):
#         print(i)
#         l,p = loss(r_temp[:,0].mean(), r_temp[:,1].mean(), r_temp[:,2].mean())
#         l_list.append(l)
#         p_list.append(p)
    torch.save([results,results_hat, l_list, p_list], 'data/results_grid'+str(alpha)+'beta100')
    print(l_list, p_list)
    
    
    
for i in range(24):
    beta_exp((i+1)/10 , 100)
    
    
    
    
## SGD


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
        w = torch.exp(self.h)/2/torch.cosh(self.h) - torch.rand(1, self.N).to(self.device)
        return sign(x.mm(w.t()))
    
    
    def exp_grad(self,x,y):
        self.sigma_update()
        sigma_sum = self.sigma.sum()
        
        temp = - (y.reshape(-1,1)* (torch.tanh(self.beta*self.h)*x).sum(1,keepdim = True)  )/(torch.sqrt(sigma_sum)+1e-10)
        #print(temp)
        #print('sigma',sigma_sum)
        #print((torch.tanh(self.beta*self.h)*x).shape)
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
        #print(grad_exp.shape, grad_KL[0,0])
        #print(grad_exp.shape)
        grad = grad_exp.sum(0,keepdim = True) +grad_KL*self.batch
        #print(grad_exp, grad_KL)
        #self.net.h_last = self.net.h*1
        self.v = self.gamma*self.v + self.lr*grad
        #print(grad[0,0], self.net.h[0][0])
        self.net.h -= self.v
        #print((self.net.h - self.net.h_last)**2)        

        
        
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
        
        
        
        
        
        




alpha_list = [(i+1)/5 for i in range(12)] + [2.1]
bs_list = [16, 32, 64, 64, 16, 16, 16, 32, 32, 32, 16, 16]
lr_list = [0.005 , 0.001 , 0.005 , 0.005 , 0.005 , 0.001 , 0.0005, 0.005 ,
        0.005 , 0.005 , 0.005 , 0.001 ]

weight_decay_list = [0.0005, 0.0005, 0.005 , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005,
        0.0005, 0.0005, 0.0005, 0.0005]



def GDexp(alpha, bs, lr, weight_decay):
    results = []
    resultss = []
    N=1000
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
        opt = Adamoptim(net, lr = lr, KL = True, beta = [0.9,0.999], weight_decay = weight_decay)
        opt.batch = batch_size/N_data
        net.h_last*=0
        t1 = time.time()
        for j in range(2000):

            for (data, label) in dataloader:
                opt.update(data,label)
                #for i in range(5):
                 #   net.h = torch.arctanh(torch.clamp(torch.tanh(net.h)/(torch.tanh(net.h)**2).mean()*0.8, -0.9999999, 0.9999999))
            '''if (j+1) % 100 == 0:
                print(1 - (torch.tanh(net.h)**2).mean(), torch.arccos((torch.sign( net.h)*teacher.h).mean())/np.pi,(torch.sign( net.h)*teacher.h).mean(), ';;')
            #print(net.h)'''
            if net.accuracy(t_data, t_label)[0] == 1:
                break
        print(time.time() - t1, bs, lr, weight_decay, torch.arccos((sign(net.h)*teacher.h).mean()).item()/np.pi )
        results.append([torch.arccos((sign(net.h)*teacher.h).mean()).item()/np.pi, (sign(net.h)*teacher.h).mean().item()])
    torch.save(results, 'data/GD_alpha='+str(alpha)+'-withKL')



for i in range(12):
    GDexp(alpha= alpha_list[i], bs = bs_list[i], lr = lr_list[i], weight_decay= weight_decay_list[i])
