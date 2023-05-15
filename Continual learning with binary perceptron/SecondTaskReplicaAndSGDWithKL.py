# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:24:38 2023

@author: huang
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time



device = torch.device('cuda')
eps = 1e-30

def H(x):
    y = -1/2*torch.erf(x/np.sqrt(2)) + 1/2
    return y

def H_prime(x):
    #print(x.shape)
    return -torch.exp(-x**2/2)/np.sqrt(2*np.pi)



def partial_E(q, r, q_star, beta, p_0 = 0):
    p_0 = 0
    z = torch.randn(10000,1).to(device).double()
    x = torch.randn(1,10000).to(device).double()
    
    g = (torch.sqrt(q_star - q + eps)*x  + torch.sqrt(q +eps)*z)/torch.sqrt(1-q_star +eps)
    D = ((H(g))**beta).mean(1, keepdim = True) + eps
    
    partial_r = H_prime( z*r/torch.sqrt((1-p_0)*q-r**2 + eps) ) * (z/torch.sqrt((1-p_0)*q-r**2 + eps) + z*r**2/((1-p_0)*q-r**2 + eps)**(3/2) ) *torch.log(D) 
    partial_r = 2*partial_r.mean()
    #print((1-p_0)*q-r**2)
    
    partial_q_1 = H_prime( z*r/torch.sqrt((1-p_0)*q-r**2 + eps) ) * ( - (1-p_0)*z*r/((1-p_0)*q-r**2 + eps)**(3/2)/2 ) *torch.log(D) 
    partial_q_2 = H( z*r/torch.sqrt((1-p_0)*q-r**2 + eps) ) * (beta * (H(g))**(beta-1) * H_prime(g) * \
                    1/2/torch.sqrt(1-q_star+eps)*(z/torch.sqrt(q + eps) - x/torch.sqrt(q_star - q + eps)) ).mean(1, keepdim = True)/D
    
    partial_q = 2*(partial_q_1.mean() + partial_q_2.mean())
    
    partial_q_star = H( z*r/torch.sqrt((1-p_0)*q-r**2 + eps) ) * (beta * (H(g))**(beta-1) * H_prime(g) * \
                    ( (x*torch.sqrt(q_star - q + eps) + torch.sqrt(q)*z)/2/(1-q_star)**(3/2) + \
                    x/2/torch.sqrt(1-q_star+eps)/torch.sqrt(q_star - q + eps)  ) ).mean(1, keepdim = True)/D

    partial_q_star = 2*partial_q_star.mean()
    
    return partial_q, partial_r, partial_q_star





def partial_S(q_1, r_1, q_1star, q_2, r_2, q_2star, k1, k, T, p_0 = 0, r_0 = 0.5):
    p_0 = 0
    z = torch.randn(500,1,1,1,1).to(device).double()
    x = torch.randn(1,500,1,1,1).to(device).double()
    m_1 = 2*torch.rand(1,1,500,1,1).to(device).double() -1
    m = 2*torch.rand(1,1,1,500,1).to(device).double() - 1
    
    partial_q2_r, partial_r2_r, partial_q2star_r, partial_k_r, partial_T_r = 0,0,0,0,0

    B_2 = torch.tensor([1,-1]).reshape(1,1,1,1,2).to(device).double()
    f1 = -(1+m_1)/2*torch.log((1+m_1)/2) - (1-m_1)/2*torch.log((1-m_1)/2)
    g = (q_1star - q_1/2)*m_1**2 + torch.sqrt(q_1 + eps)*z*m_1 + r_1*m_1 + k1*f1
    f = -(1+m)*torch.log( (1 + m)/(1 + m_1) )/2 - (1-m)*torch.log( (1-m) / (1-m_1 ))/2
    h = (q_2star - q_2/2)*m**2 +torch.sqrt(q_2 + eps)*x*m + k*f + r_2*m*B_2 + T*m
    D = torch.exp(g).mean(2, keepdim = True)*2 + eps
    
    
    
    M1 = (torch.exp(h) * (-m**2/2 + x*m /2/ torch.sqrt(q_2 + eps))).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True)+ eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_q2 = (M2/D)
    #print(partial_q2[:,:,:,:,1].mean())
    partial_q2 = partial_q2[:,:,:,:,0].mean()*(1-p_0)**2/2*(1+r_0) + partial_q2[:,:,:,:,1].mean()*(1-p_0)**2/2*(1-r_0)
    
    
    
    M1 = (torch.exp(h) * (m*B_2)).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True) + eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_r2 = (M2/D)
    partial_r2 = partial_r2[:,:,:,:,0].mean()*(1-p_0)**2/2*(1+r_0) + partial_r2[:,:,:,:,1].mean()*(1-p_0)**2/2*(1-r_0)
    
    M1 = (torch.exp(h) * (m)**2).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True) + eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_q2star = (M2/D)
    partial_q2star = partial_q2star[:,:,:,:,0].mean()*(1-p_0)**2/2*(1+r_0) + partial_q2star[:,:,:,:,1].mean()*(1-p_0)**2/2*(1-r_0)
    
    
    
    M1 = (torch.exp(h) * (m)).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True) + eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_T = (M2 / D)
    partial_T = partial_T[:,:,:,:,0].mean()*(1-p_0)**2/2*(1+r_0) + partial_T[:,:,:,:,1].mean()*(1-p_0)**2/2*(1-r_0)
    
    partial_q2_r += partial_q2
    partial_r2_r += partial_r2
    partial_q2star_r += partial_q2star
    
    partial_T_r += partial_T
    
    
    
    return partial_q2_r, partial_r2_r, partial_q2star_r, partial_k_r, partial_T_r





def exp(alpha_2, beta_2, q_1_hat, r_1_hat, q_1star_hat, r_0):
    results = []
    beta1 = 100
    for times in range(5):
        
        k = -torch.rand(1).to(device).double()
        T = torch.rand(1).to(device).double()
        temp_ = 0

        q_2 =  0.1*torch.rand(1).to(device).double()
        r_2 = 0.1*torch.rand(1).to(device).double()*q_2
        q_2star = 0.1*torch.rand(1).to(device).double()*q_2 + q_2
        k = -torch.rand(1).to(device).double()
        T = torch.rand(1).to(device).double()
        for i in range(100):

            temp = torch.tensor([ q_2, r_2, q_2star])        
            temp1 = torch.zeros([20, 3]).double().to(device)
            j = 0
            k = 0
            n_fails = 0
            while(j<20):
                temp1[j,0],temp1[j,1],temp1[j,2] =  partial_E(q_2, r_2, q_2star, beta_2, p_0 = 0)
                if not (temp1[j]!=temp1[j]).sum():
                    print
                    j+=1
                    n_fails = 0
                else:
                    n_fails += 1

                if n_fails > 5:



                    q_2 =  0.1*torch.rand(1).to(device).double()
                    r_2 = 0.1*torch.rand(1).to(device).double()*q_2
                    q_2star = 0.1*torch.rand(1).to(device).double()*q_2 + q_2


                    k = -torch.rand(1).to(device).double()
                    T = torch.rand(1).to(device).double()
                    n_fails = 0
                    print('change init', end = 'emmm.....')
                k += 1

                if k>100:
                    print('not complete!',j,end='=e')
                    break



            q_2_hat, r_2_hat, q_2star_hat = temp1[:j,:].mean(0)


            q_2_hat *= (-2*alpha_2)
            r_2_hat *= alpha_2
            q_2star_hat *= alpha_2
            k_hat = beta_2
            T_hat = 0
            print('.....')
            j = 0
            k = 0
            n_fails = 0
            temp2 = torch.zeros(20,5).double()
            while(j<20):
                temp2[j, 0], temp2[j, 1], temp2[j, 2], temp2[j, 3], temp2[j, 4],  = \
                                        partial_S(q_1_hat, r_1_hat, q_1star_hat,q_2_hat, r_2_hat, q_2star_hat,beta1, k_hat, T_hat, p_0 = 0, r_0 = r_0)

                if not (temp2[j]!=temp2[j]).sum():
                    j+=1     
                k += 1
                if k>50:
                    print('not complete!',j,end='=j')
                    break



            q_2, r_2, q_2star, k, T = temp2[:j,:].mean(0)
            q_2 *= (-2)
            #print('ddds', q_1,  r_1, q_1star, q_2, r_2, q_2star, k, T)
            err = ((torch.tensor([q_2, r_2, q_2star]) - temp)**2).mean()
            print(err, end = ';;')
            if (i == 99) or ( err<1e-5):
                #results.append([ q_2.item(), r_2.item(), q_2star.item(), k,T])
                #results_hat.append([ q_2_hat.item(), r_2_hat.item(), q_2star_hat.item(), ])
                print('alpha_1',alpha_2,'beta',beta_2, ';',i,';',err,';;',q_2.item(), r_2.item(), q_2star.item(),q_2_hat.item(), r_2_hat.item(), q_2star_hat.item(),r_0)
                #torch.save([ q_2.item(), r_2.item(), q_2star.item(), k,T, q_2_hat.item(), r_2_hat.item(), q_2star_hat.item()], 'secondtask_beta='+str(beta_2)+'_alpha='+str(alpha_2)+'_time='+str(times)+'q=0.5')
                
                break
        results.append([q_2.item(), r_2.item(), q_2star.item(), k,T, q_2_hat.item(), r_2_hat.item(), q_2star_hat.item(),r_0])
    torch.save(results, 'data/second-task-'+'alpha-'+str(alpha_2) + 'r_0-'+(str(r_0))+'-beta1=100-task1withKL')
    
_,results_hat,_,_ = torch.load('data/results_grid'+str(20/10))
q_1_hat, r_1_hat, q_1star_hat = torch.tensor(results_hat).cuda().mean(0).double()
r_0_list = [-0.5,0,0.5]
alpha_list = [5*(i+1)/10 for i in range(10)]
for r_0 in r_0_list:
    for alpha_2 in alpha_list:
        exp(alpha_2, 20, q_1_hat, r_1_hat, q_1star_hat, r_0)
        
        
        
        










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
        w = torch.exp(self.h)/2/torch.cosh(self.h) - torch.rand(1, self.N).cuda()
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
        
        
        
def GDexp_second_task(alpha_1, alpha_2, r_0, lr, bs, weight_decay = 1e-3):
    results = []
    resultss = []
    for timesd in range(20):
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
        dataloader_1 = torch.utils.data.DataLoader(data_list_1, batch_size = 32, shuffle = False)
        dataloader_2 = torch.utils.data.DataLoader(data_list_2, batch_size = bs, shuffle = False)

        net = Perceptron(beta = 1, N = N,device = device)

        opt = Adamoptim(net, lr = 5e-3, KL = True, beta = [0.9,0.999], weight_decay = 5e-4)
        opt.batch = 32/N_data_1
        net.h_last*=0

        for j in range(1000):
            for (data, label) in dataloader_1:
                opt.update(data,label)
            #if (j+1) % 100 == 0:
              #  print(1 - (torch.tanh(net.h)**2).mean(),net.accuracy_tri(t_data, t_label)[0], (double_step(torch.tanh(net.h)) == torch.sign(teacher.h)).float().mean(),  net.accuracy_tri(t_data_2, t_label_2)[0], (double_step(torch.tanh(net.h)) == torch.sign(teacher2.h)).float().mean(), ';;')
        #print(torch.arccos(((torch.sign(net.h))* torch.sign(teacher.h)).float().mean())/np.pi, torch.arccos(((torch.sign(net.h)) *torch.sign(teacher2.h)).float().mean())/np.pi)
        opt = Adamoptim(net, lr = lr, KL = True, beta = [0.9,0.999], weight_decay = weight_decay)
        opt.batch = bs/N_data_2
        net.h_last = net.h*1
        for j in range(3000):
            for (data, label) in dataloader_2:
                opt.update(data,label)
            if np.arccos((sign(net.h)*teacher2.h).mean().item())/np.pi ==0:
                break
            if j %100 ==0:
                print(alpha_1,alpha_2, r_0, lr, bs,np.arccos((sign(net.h)*teacher2.h).mean().item())/np.pi)
        results.append([np.arccos((sign(net.h)*teacher.h).mean().item())/np.pi, (sign(net.h)*teacher.h).mean().item(), np.arccos((sign(net.h)*teacher2.h).mean().item())/np.pi, (sign(net.h)*teacher2.h).mean().item()])
    torch.save(results, 'data/GD_alpha_second='+str(alpha_1) +','+str(alpha_2)+'r_0='+str(r_0)+'-task1KL')
    

    


alpha_2_list = [(i+1)/2 for i in range(9)]
r_0_list = [-0.5,0,0.5] 
lr_list = np.array([[0.005, 0.005, 0.001, 0.01 , 0.005, 0.001, 0.01 , 0.01 , 0.001],
        [0.001, 0.001, 0.01 , 0.005, 0.01 , 0.01 , 0.005, 0.001, 0.001],
        [0.005, 0.001, 0.005, 0.01 , 0.01 , 0.001, 0.001, 0.001, 0.001]])

bs_list = np.array([[32., 16., 32., 16., 16., 16., 16., 16., 16.],
        [16., 16., 32., 16., 16., 16., 16., 16., 16.],
        [64., 16., 32., 16., 16., 16., 16., 16., 16.]])

for i in range(len(alpha_2_list)):
    
    for j in range(len(r_0_list)):
        
        GDexp_second_task(alpha_1=2, alpha_2 = alpha_2_list[i],
                          r_0 = r_0_list[j], lr = lr_list[j,i], 
                          bs = bs_list[j,i] , weight_decay = 1e-3)