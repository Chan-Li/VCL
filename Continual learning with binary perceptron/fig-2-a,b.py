# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:04:17 2023

@author: huang
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
#from scipy import integrate
import matplotlib as mpl
import os
os.system('python FirstTaskReplica.py')
os.system('python FirstTaskSGD.py')


device = torch.device('cuda')
eps = 1e-50

def H(x):
    y = -1/2*torch.erf(x/np.sqrt(2)) + 1/2
    return y

def H_prime(x):
    return -torch.exp(-x**2/2)/np.sqrt(2*np.pi)



def loss(q, r, q_star, beta, largealpha = True):
    k = beta
    z = torch.randn(10000,1).to(device).double()
    partial_p  = 0
    if not largealpha:
        
        m = 2*torch.rand(1,10000).to(device).double() -1
        h = (q_star - q/2 + eps)*m**2 + torch.sqrt(q + eps)*z*m + r*m 
        h = h - h.max()
        D = torch.exp(h).mean(1,keepdim = True) + eps
        M = torch.exp(h)
        intterm = M/D
        intterm[D.reshape(-1)==0,:] = 0
       
        p = (( torch.sign(m)  )*intterm).mean()
        l = torch.arccos(p)/np.pi
        
    else:
        D = 0
        for i in range(20):
            phi = torch.rand(1,10000).to(device).double()
            m = (torch.log(torch.exp( torch.sqrt(q)*z + r -400)* phi +  (1-phi)* torch.exp(-torch.sqrt(q)*z - r-400)) + 400 )/(torch.sqrt(q)*z + r)
            h = (q_star - q/2 + eps)*m**2  
            h = h-h.max()
            D += torch.exp(h).mean(1,keepdim = True) 
            M = torch.exp(h)
            partial_p += (( torch.sign(m)  )*M).mean(1, keepdim = True)
            
       
        partial_p = (partial_p/(D)).mean()
        
        p = partial_p       
        l = torch.arccos(p)/np.pi
    
    return l,p



l_GD = []
p_GD = []
for i in range(17):
    a = torch.load('data/GD_alpha='+str((i+1)/10))
    l_GD.append([a[i][0] for i in range(len(a))])
    p_GD.append([a[i][1] for i in range(len(a))])
l_GD = np.array(l_GD)
p_GD = np.array(p_GD)
l_GD = np.arccos(p_GD)/np.pi

results =[]
results_hat = []
l_list = []
p_list = []
for i in range(200):
    a,b = torch.load('data/results_grid'+str((i+1)/100))
    results.append(a)
    results_hat.append(b)
results = np.array(results)
results_hat = np.array(results_hat)


alpha_list = [(i+1)/100 for i in range(200)]
l_list = []
p_list = []
t1 = time.time()
for j in range(50):
    t2 = time.time()
    print(j, t2-t1, end = ';')
    t1 = t2
    l1, p =[], []
    for i in range(len(alpha_list)):
        if i >150:
            largealpha = True
        else:
            False
        
        hat = results_hat[i].mean(0)
        temp = torch.from_numpy(hat).cuda()
        a,b = loss(temp[0], temp[1], temp[2], 20,largealpha = True)
        l1.append(a.item())
        p.append(b.item())
    l_list.append(l1)
    p_list.append(p)
l = np.array(l_list)
p = np.array(p_list)



color = ['black','dimgrey','darkgray']
label = [r'$q_0$', r'$r_1$', r'$q_d$']
linestlye = ['-','--', ':']



axis = plt.subplot(111)
for i in range(3):
    plt.plot([(i+1)/100 for i in range(200)], results[:,:,i].mean(1),'--', linestyle=linestlye[i], color = color [i], label = label[i], linewidth = 3)
plt.legend(fontsize = 15)
plt.xlabel(r'$\alpha$', fontsize = 30)
plt.tick_params(which = 'major', direction = 'in', width = 4, length = 8)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.savefig('fig/fig-2-a.pdf', bbox_inches = 'tight')




plt.figure(figsize = (6,5))
axis = plt.subplot(111)
plt.plot([(i+1)/100 for i in range(200)], p, color = color[0],label = r'$p_1$', linewidth = 3)
plt.errorbar([(i+1)/10 for i in range(17)],p_GD.mean(1),p_GD.std(1),capsize = 3,markerfacecolor='white',marker = 'o', linestyle = '--', color = color[0], label = r'$p_1^{SGD}$', linewidth = 2 )
plt.plot([(i+1)/100 for i in range(200)], l,  color = color[1], label = r'$\epsilon_g$', linewidth = 3)
plt.errorbar([(i+1)/10 for i in range(17)],l_GD.mean(1), l_GD.std(1),capsize = 3,markerfacecolor='white',marker = 's',linestyle = '--', color = color[1], label = r'$\epsilon_g^{SGD}$', linewidth = 2 )
plt.legend(fontsize = 15)
plt.text(1.5, 0.67, r"$\epsilon_g\propto\alpha^{-13.1}$", size=15)

plt.xlabel(r'$\alpha$', fontsize = 30)
plt.tick_params(which = 'major', direction = 'in', width = 4, length = 8)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
plt.savefig('fig/fig-2-b.pdf', bbox_inches = 'tight')




