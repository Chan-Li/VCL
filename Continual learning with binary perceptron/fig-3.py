# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:59:42 2023

@author: huang
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
#from scipy import integrate
import matplotlib as mpl
import os
os.system('python SeccondTaskReplica.py')
os.system('python SecondTaskGD.py')
device = torch.device('cuda')
eps = 1e-50

def H(x):
    #print(x[0])
    #print(-1/2*torch.erf(x/np.sqrt(2))[0] + 1/2)
    y = -1/2*torch.erf(x/np.sqrt(2)) + 1/2
    #print(x.shape)
    #y = torch.randn(x.shape[0],10000).cuda()
    #y[y<x] = 0
    #y = y.sum(1,keepdim = True)/10000
    return y

def H_prime(x):
    #print(x.shape)
    return -torch.exp(-x**2/2)/np.sqrt(2*np.pi)








def Loss(q_1, r_1, q_1star, q_2, r_2, q_2star, k, T, p_0 = 0, r_0 = 0):
    z = torch.randn(500,1,1,1,1).to(device).double()
    x = torch.randn(1,500,1,1,1).to(device).double()
    m_1 = 2*torch.rand(1,1,500,1,1).to(device).double() -1
    m = 2*torch.rand(1,1,1,500,1).to(device).double() - 1
    
    partial_p1_r, partial_p2_r = 0,0,
    ## 两个任务都不为0
    B_2 = torch.tensor([1,-1]).reshape(1,1,1,1,2).to(device).double()
    g = (q_1star - q_1/2)*m_1**2 + torch.sqrt(q_1 + eps)*z*m_1 + r_1*m_1
    f = -(1+m)*torch.log( (1 + m)/(1 + m_1) )/2 - (1-m)*torch.log( (1-m) / (1-m_1 ))/2
    h = (q_2star - q_2/2)*m**2 +torch.sqrt(q_2 + eps)*x*m + k*f + r_2*m*B_2 + T*m
    D = torch.exp(g).mean(2, keepdim = True)*2 + eps
    
    
    
    M1 = (torch.exp(h) * (torch.sign(m)) ).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True)+ eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_p1 = (M2/D)
    partial_p1 = partial_p1[:,:,:,:,0].mean()*(1-p_0)**2/2*(1+r_0) + partial_p1[:,:,:,:,1].mean()*(1-p_0)**2/2*(1-r_0)
    
    M1 = (torch.exp(h) * (torch.sign(m)*B_2)).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True) + eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_p2 = (M2/D)
    partial_p2 = partial_p2[:,:,:,:,0].mean()*(1-p_0)**2/2*(1+r_0) + partial_p2[:,:,:,:,1].mean()*(1-p_0)**2/2*(1-r_0)
    
    partial_p1_r += partial_p1
    partial_p2_r += partial_p2
    
    l_1 = torch.arccos(partial_p1_r /torch.sqrt(torch.tensor(1).to(device)-p_0))/np.pi
    l_2 = torch.arccos(partial_p2_r/torch.sqrt(torch.tensor(1).to(device)-p_0))/np.pi
    
    return partial_p1_r, partial_p2_r, l_1,l_2 






_,results_hat,_,_ = torch.load('data/results_grid'+str(2)+'beta'+str(20))
q_1_hat, r_1_hat, q_1star_hat = torch.tensor(results_hat).cuda().mean(0).double()
r_0_list = [-0.5,0,0.5]
alpha_list = [0.01,0.1 ] +[5*(i+1)/10 for i in range(10)] 
results = []
for r_0 in r_0_list:
    result = []
    for alpha_2 in alpha_list:
        if alpha_2 == 9.0 and r_0 == 0.5:
            result.append([[torch.nan for j in range(9)] for i in range(5) ])
        else:
            result.append( torch.load( 'data/second-task-'+'alpha-'+str(alpha_2) + 'r_0-'+(str(r_0)) )  )
        #print(result[-1])
    results.append(result)
results = torch.tensor(results).cuda()





q_2 = results[:,:,:,0].mean(2) 
r_2 = results[:,:,:,1].mean(2) 
q_2star = results[:,:,:,2].mean(2) 
q_2_hat = results[:,:,:,5].mean(2) 
r_2_hat = results[:,:,:,6].mean(2) 
q_2star_hat = results[:,:,:,7].mean(2) 

l_1 = torch.zeros(q_2.shape).double().cuda()
l_2 = torch.zeros(q_2.shape).double().cuda()
p_1 = torch.zeros(q_2.shape).double().cuda()
p_2 = torch.zeros(q_2.shape).double().cuda()
r_0_list = [-0.5,0,0.5]
for i in range(q_2.shape[0]):
    for j in range(q_2.shape[1]):
        l1 = []
        l2 = []
        p1 = []
        p2 = []
        for k in range(30):
            a,b,c,d = Loss(q_1_hat, r_1_hat, q_1star_hat, q_2_hat[i,j], r_2_hat[i,j], q_2star_hat[i,j], 20, 0, p_0 = 0, r_0 = r_0_list[i])
            if (a==a) and (b==b) and (c==c) and (d==d):
                p1.append(a)
                p2.append(b)
                l1.append(c)
                l2.append(d)
        l_1[i,j] = torch.tensor(l1).mean()
        l_2[i,j] = torch.tensor(l2).mean()
        p_1[i,j] = torch.tensor(p1).mean()
        p_2[i,j] = torch.tensor(p2).mean()
        print(i,j)
        

        




r_0_list = [-0.5,0,0.5]
alpha_list = [0.1] + [5*(i+1)/10 for i in range(13)]
l_1_GD = []
l_2_GD = []
p_1_GD = []
p_2_GD = []
alpha_1 = 2
for r_0 in r_0_list:
    l1_GD = []
    l2_GD = []
    p1_GD = []
    p2_GD = []
    for alpha_2 in alpha_list:
        if r_0 >-0.1 and alpha_2 >6.1:
            a = np.array([[np.nan for i in range(4)] for k in range(20)])
        else:
            a = torch.load('data/GD_alpha_second='+str(alpha_1) +','+str(alpha_2)+'r_0='+str(r_0))
            a = np.array(a)
        
        l1_GD.append(a[:, 0])
        p1_GD.append(a[:, 1])
        l2_GD.append(a[:, 2])
        p2_GD.append(a[:, 3])
    l_1_GD.append(l1_GD)
    l_2_GD.append(l2_GD)
    p_1_GD.append(p1_GD)
    p_2_GD.append(p2_GD)

l_1_GD = np.array(l_1_GD)
l_2_GD = np.array(l_2_GD)
p_1_GD = np.array(p_1_GD)
p_2_GD = np.array(p_2_GD)





        
label = [r'$q_0$', r'$r_2$', r'$q_{0}$']
label_r = [r'$r_0=-0.5$', r'$r_0=0$', r'$r_0=0.5$']
color = ['r', 'b', 'black']
color1 = ['salmon', 'royalblue', 'dimgray']
marker = ['o', 's', '^']
plt.figure(figsize = (6,5))
axis = plt.subplot(111)
for i in range(3):
    
    plt.plot([0.01,0.1 ] +[(j+1)/2 for j in range(14)] , l_2[i].cpu(),'-',marker = marker[i],markersize = 8, markerfacecolor = "None", label = label_r[i], color = color[i])
    if i ==0:
        plt.errorbar([0.1] + [5*(i+1)/10 for i in range(13)], l_2_GD[i].mean(1), l_2_GD[i].std(1),linestyle = '--', 
                     capsize = 3,markersize = 8,markerfacecolor = "None",marker = marker[i], color = color1[i])
    else:
        plt.errorbar([0.1] + [5*(i+1)/10 for i in range(12)], l_2_GD[i].mean(1)[:-1], l_2_GD[i].std(1)[:-1],linestyle = '--', 
                     capsize = 3,markersize = 8,markerfacecolor = "None",marker = marker[i], color = color1[i])
    #plt.plot([(i+1)/2 for i in range(10)], l_1[i].cpu(), label = label_r[i])
plt.legend(fontsize = 15)
plt.xlabel(r'$\alpha_2$', fontsize = 30)
plt.ylabel(r'$\epsilon_g^2$', fontsize = 30)
plt.title(r'$\gamma = 1$', fontsize = 30)
plt.tick_params(which = 'major', direction = 'in', width = 4, length = 8)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
#plt.savefig('fig/fig-1-2.pdf', bbox_inches = 'tight')
plt.savefig('fig/fig-3-a.pdf', bbox_inches = 'tight')




label = [r'$q_2$', r'$r_2$', r'$q_{2,\star}$']
label_r = [r'$r_0=-0.5$', r'$r_0=0$', r'$r_0=0.5$']
color = ['r', 'b', 'black']
color1 = ['salmon', 'royalblue', 'dimgray']
plt.figure(figsize = (6,5))
axis = plt.subplot(111)
for i in range(3):
    plt.plot([0.01,0.1 ] +[(j+1)/2 for j in range(14)], l_1[i].cpu() , '-',marker = marker[i], markersize = 8,markerfacecolor = "None", label = label_r[i], color = color[i])
    if i ==0:
        plt.errorbar([0.1]+[(i+1)/2 for i in range(13)], l_1_GD[i].mean(1), l_1_GD[i].std(1),marker = marker[i],linestyle = '--', 
                 capsize = 3,markersize = 8,markerfacecolor = "None",color = color1[i])
    else:
        plt.errorbar([0.1]+[(i+1)/2 for i in range(12)], l_1_GD[i].mean(1)[:-1], l_1_GD[i].std(1)[:-1],marker = marker[i],linestyle = '--', 
                 capsize = 3,markersize = 8,markerfacecolor = "None",color = color1[i])
    #plt.plot([(i+1)/2 for i in range(10)], l_1[i].cpu(), label = label_r[i])
plt.legend(fontsize = 15)
plt.xlabel(r'$\alpha_2$', fontsize = 30)
plt.ylabel(r'$\epsilon_g^1$', fontsize = 30)
plt.title(r'$\gamma = 1$', fontsize = 30)
plt.tick_params(which = 'major', direction = 'in', width = 4, length = 8)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
plt.savefig('fig/fig-3-b.pdf', bbox_inches = 'tight')




# gamma = 0.1

gamma =0.1

def Loss(q_1, r_1, q_1star, q_2, r_2, q_2star, k, T, p_0 = 0, r_0 = 0):
    z = torch.randn(500,1,1,1,1).to(device).double()
    x = torch.randn(1,500,1,1,1).to(device).double()
    m_1 = 2*torch.rand(1,1,500,1,1).to(device).double() -1
    m = 2*torch.rand(1,1,1,500,1).to(device).double() - 1
    
    partial_p1_r, partial_p2_r = 0,0,
    ## 两个任务都不为0
    B_2 = torch.tensor([1,-1]).reshape(1,1,1,1,2).to(device).double()
    g = (q_1star - q_1/2)*m_1**2 + torch.sqrt(q_1 + eps)*z*m_1 + r_1*m_1
    f = -(1+m)*torch.log( (1 + m)/(1 + m_1) )/2 - (1-m)*torch.log( (1-m) / (1-m_1 ))/2
    f*=gamma
    h = (q_2star - q_2/2)*m**2 +torch.sqrt(q_2 + eps)*x*m + k*f + r_2*m*B_2 + T*m
    D = torch.exp(g).mean(2, keepdim = True)*2 + eps
    
    
    
    M1 = (torch.exp(h) * (torch.sign(m)) ).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True)+ eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_p1 = (M2/D)
    partial_p1 = partial_p1[:,:,:,:,0].mean()*(1-p_0)**2/2*(1+r_0) + partial_p1[:,:,:,:,1].mean()*(1-p_0)**2/2*(1-r_0)
    
    M1 = (torch.exp(h) * (torch.sign(m)*B_2)).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True) + eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_p2 = (M2/D)
    partial_p2 = partial_p2[:,:,:,:,0].mean()*(1-p_0)**2/2*(1+r_0) + partial_p2[:,:,:,:,1].mean()*(1-p_0)**2/2*(1-r_0)
    
    partial_p1_r += partial_p1
    partial_p2_r += partial_p2
    
    
    
    
    partial_p1_r += partial_p1
    partial_p2_r += partial_p2
    
    l_1 = torch.arccos(partial_p1_r /torch.sqrt(torch.tensor(1).to(device)-p_0))/np.pi
    l_2 = torch.arccos(partial_p2_r/torch.sqrt(torch.tensor(1).to(device)-p_0))/np.pi
    
    return partial_p1_r, partial_p2_r, l_1,l_2 






_,results_hat,_,_ = torch.load('results_grid'+str(2)+'beta'+str(20))
q_1_hat, r_1_hat, q_1star_hat = torch.tensor(results_hat).cuda().mean(0).double()
r_0_list = [-0.5,0,0.5]
alpha_list = [0.01,0.1,0.5,1,2,5]
results = []
for r_0 in r_0_list:
    result = []
    for alpha_2 in alpha_list:
        result.append( torch.load( 'data/second-task-'+'alpha-'+str(alpha_2) + 'r_0-'+(str(r_0)) + 'smallKL' + '0.1')  )
    results.append(result)
results = torch.tensor(results).cuda()



q_2 = results[:,:,:,0].mean(2)
r_2 = results[:,:,:,1].mean(2)
q_2star = results[:,:,:,2].mean(2)
q_2_hat = results[:,:,:,5].mean(2)
r_2_hat = results[:,:,:,6].mean(2)
q_2star_hat = results[:,:,:,7].mean(2)
l_1 = torch.zeros(q_2.shape).double().cuda()
l_2 = torch.zeros(q_2.shape).double().cuda()
p_1 = torch.zeros(q_2.shape).double().cuda()
p_2 = torch.zeros(q_2.shape).double().cuda()
r_0_list = [-0.5,0,0.5]
for i in range(q_2.shape[0]):
    for j in range(q_2.shape[1]):
        l1 = []
        l2 = []
        p1 = []
        p2 = []
        for k in range(30):
            a,b,c,d = Loss(q_1_hat, r_1_hat, q_1star_hat, q_2_hat[i,j], r_2_hat[i,j], q_2star_hat[i,j], 20, 0, p_0 = 0, r_0 = r_0_list[i])
            if (a==a) and (b==b) and (c==c) and (d==d):
                p1.append(a)
                p2.append(b)
                l1.append(c)
                l2.append(d)
        l_1[i,j] = torch.tensor(l1).mean()
        l_2[i,j] = torch.tensor(l2).mean()
        p_1[i,j] = torch.tensor(p1).mean()
        p_2[i,j] = torch.tensor(p2).mean()
        print(i,j)
        
        



r_0_list = [-0.5,0,0.5]
alpha_list = [0.01, 0.1, 0.5, 1,2,5]
l_1_GD = []
l_2_GD = []
p_1_GD = []
p_2_GD = []
alpha_1 = 2
for r_0 in r_0_list:
    l1_GD = []
    l2_GD = []
    p1_GD = []
    p2_GD = []
    for alpha_2 in alpha_list:
        if r_0 >-0.1 and alpha_2 >6.1:
            a = np.array([[np.nan for i in range(4)] for k in range(20)])
        else:
            #'GD_alpha_second='+str(alpha_1) +','+str(alpha_2)+'r_0='+str(r_0)+'gamma=0.1'
            a = torch.load('data/GD_alpha_second='+str(alpha_1) +','+str(alpha_2)+'r_0='+str(r_0)+'gamma=0.1')
            a = np.array(a)
        
        l1_GD.append(a[:, 0])
        p1_GD.append(a[:, 1])
        l2_GD.append(a[:, 2])
        p2_GD.append(a[:, 3])
    l_1_GD.append(l1_GD)
    l_2_GD.append(l2_GD)
    p_1_GD.append(p1_GD)
    p_2_GD.append(p2_GD)

l_1_GD = np.array(l_1_GD)
l_2_GD = np.array(l_2_GD)
p_1_GD = np.array(p_1_GD)
p_2_GD = np.array(p_2_GD)



label = [r'$q_2$', r'$r_2$', r'$q_{2,\star}$']
label_r = [r'$r_0=-0.5$', r'$r_0=0$', r'$r_0=0.5$']
color = ['r', 'b', 'black']
color1 = ['salmon', 'royalblue', 'dimgray']
marker = ['o', 's', '^']
plt.figure(figsize = (6,5))
axis = plt.subplot(111)
for i in range(3):
    plt.plot([0.01, 0.1,0.5,1,2,5], l_1[i].cpu() , '-',marker = marker[i], markersize = 8,markerfacecolor = "None", label = label_r[i], color = color[i])

    plt.errorbar([0.01, 0.1,0.5,1,2,5], l_1_GD[i].mean(1), l_1_GD[i].std(1),marker = marker[i],linestyle = '--', 
                 capsize = 3,markersize = 8,markerfacecolor = "None",color = color1[i])
    #plt.plot([(i+1)/2 for i in range(10)], l_1[i].cpu(), label = label_r[i])
plt.semilogx()
plt.legend(fontsize = 15)
plt.xlabel(r'$\alpha_2$', fontsize = 30)
plt.ylabel(r'${\epsilon}_g^1$', fontsize = 30)
plt.tick_params(which = 'major', direction = 'in', width = 4, length = 8)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
plt.title(r'$\gamma$ = 0.1', fontsize = 30)
plt.savefig('fig/fig-3-c.pdf', bbox_inches = 'tight')




label = [r'$q_2$', r'$r_2$', r'$q_{2,\star}$']
label_r = [r'$r_0=-0.5$', r'$r_0=0$', r'$r_0=0.5$']
color = ['r', 'b', 'black']
color1 = ['salmon', 'royalblue', 'dimgray']
marker = ['o', 's', '^']
plt.figure(figsize = (6,5))
axis = plt.subplot(111)
for i in range(3):
    plt.plot([0.01, 0.1,0.5,1,2,5], l_2[i].cpu() , '-',marker = marker[i], markersize = 8,markerfacecolor = "None", label = label_r[i], color = color[i])

    plt.errorbar([0.01, 0.1,0.5,1,2,5], l_2_GD[i].mean(1), l_2_GD[i].std(1),marker = marker[i],linestyle = '--', 
                 capsize = 3,markersize = 8,markerfacecolor = "None",color = color1[i])
    #plt.plot([(i+1)/2 for i in range(10)], l_1[i].cpu(), label = label_r[i])
plt.semilogx()
plt.legend(fontsize = 15)
plt.xlabel(r'$\alpha_2$', fontsize = 30)
plt.ylabel(r'${\epsilon}_g^2$', fontsize = 30)
plt.tick_params(which = 'major', direction = 'in', width = 3, length = 6)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
#axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
plt.title(r'$\gamma$ = 0.1',  fontsize = 30)
#plt.ylim(0.47,0.52)
plt.savefig('fig/fig-3-d.pdf', bbox_inches = 'tight')