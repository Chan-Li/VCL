import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F


#%% functions
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
    def __init__(self, N = 1000, beta = 1, device = torch.device('cuda')):
        self.beta = beta
        self.h = (torch.randn(1,N)).to(device)
        self.w = sign(self.h)
        self.sigma = 1-torch.tanh(self.beta*self.h)**2
        self.device = device
        self.N = N

    def sigma_update(self):
        self.sigma =  1-torch.tanh(self.beta*self.h)**2
        
    def forward(self,x):
        w = torch.sign(self.h)
        return sign(x.mm(w.T))
    
    def exp_grad(self, x, y):
        # x shape : [M, N], grad shape : [M, N]
        self.sigma_update()
        sigma_sum = self.sigma.sum()
        temp = -(y.reshape(-1,1)*(torch.tanh(self.beta*self.h)*x).sum(1,keepdim=True))\
                /(torch.sqrt(sigma_sum)+1e-10)
        grad = self.beta*self.sigma * (y.reshape(-1,1)/(sigma_sum**(3/2)+1e-10)\
                * ( x*sigma_sum + torch.tanh(self.beta * self.h)\
                * (torch.tanh(self.beta*self.h)*x).sum(1,keepdim = True))\
                * H_prime(temp)/(H(temp) +1e-10))    
        return grad

    def accuracy(self, x, y):
        # y int variable
        yhat = self.forward(x).reshape(-1)
        acc = (yhat==y).mean()
        return acc
    

class SGDoptim():
    def __init__(self, net, lr = 1e-3, batch = 1, device = torch.device('cuda')):
        self.net = net
        self.lr = lr
        self.batch = batch
        
    def update(self, x, y):
        batch_size =  x.shape[0] // self.batch
        for batch_ind in range(batch):
            x_batch = x[batch_ind*batch_size:(batch_ind+1)*batch_size,:]*1
            y_batch = y[batch_ind*batch_size:(batch_ind+1)*batch_size]*1
            grad_exp = self.net.exp_grad(x_batch*1, y_batch*1)
            grad = grad_exp.sum(0, keepdim = True)
            self.net.h -= self.lr*grad 
            
        if batch_size*self.batch < x.shape[0]:
            x_batch = x[batch_size*batch:,:]*1
            y_batch = x[batch_size*batch:]*1
            grad_exp = self.net.exp_grad(x_batch*1, y_batch*1)
            grad = grad_exp.sum(0, keepdim = True)
            self.net.h -= self.lr*grad 


#%% Batch Run
N = 3000
alpha = 1.6
lr = 0.01

SGD_dicts = {}
for alpha in np.arange(1.0, 2.02, 0.02):
    SGD_dicts[alpha] = {}
    M = int(N*alpha)
    for r in range(20):
        print("alpha: {}, sample: {}".format(alpha, r))
        
        batch = 1
        data = torch.from_numpy(np.random.choice([-1,1], size = [M, N]))\
                .to(torch.device('cuda')).float()
        teacher = Perceptron()
        teacher.h = torch.from_numpy(np.random.choice([-1,1], size = [1, N]))\
                .to(torch.device('cuda')).float()        
        label = teacher.forward(data).reshape(-1)
        
        # test 
        t_data = torch.from_numpy(np.random.choice([-1,1], size = [10000, N]))\
                .to(torch.device('cuda')).float()
        t_label = teacher.forward(t_data).reshape(-1)
        
        # train
        net = Perceptron()
        net.h = torch.randn(1,N).to(torch.device('cuda'))
        opt = SGDoptim(net, lr = lr, batch = batch )
        
        t1 = time.time()
        overlaps = []
        for j in range(10000):
            opt.update(data, label)
            overlap = (torch.sign(net.h)*teacher.h).mean()
            overlaps.append(overlap.cpu().numpy())


        SGD_dicts[alpha][r] = np.array(overlaps*1)



#%% Results
SGD_dicts_load = np.load("SGD_dict.npy", allow_pickle=True).item()

def converge_time(p):
    # p -> error
    loss = np.arccos(p)/np.pi
    data = np.copy(loss)
    
    # find the Tc
    T_end = 10000
    T = 1000
    shr = 0.0005
    varis = np.zeros(T_end-T)
    for t in np.arange(T, T_end):
        data_T = data[t-T:t]
        varis[t-T] = data_T.std()
    for t in range(T_end-T):
        if varis[t]<shr:
            break
    tc = t + T
    return tc

tcs_array = np.zeros((51, 20))
for i,alpha in enumerate(np.arange(1.0, 2.02, 0.02)):
    data_alpha = SGD_dicts_load["{:.2f}".format(alpha)]
    for r in range(20):
        tc = converge_time(data_alpha[r])
        tcs_array[i, r] = tc

# np.save('tc_array.npy', tcs_array)
#%% Figure
# data 
alpha = np.arange(1.0, 2.02, 0.02)
tc_mean, tc_std =  tcs_array.mean(1), tcs_array.std(1)

# figure
plt.figure(figsize=(5,4), dpi=300)
plt.plot(alpha, tc_mean, c='royalblue', lw=2,  ls='--')
plt.fill_between(alpha, tc_mean+tc_std, tc_mean-tc_std, color='royalblue', alpha=0.5)
plt.xlabel(r'$\alpha$', size=15)
plt.ylabel(r'$T_c$', size=15)
