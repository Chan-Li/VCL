# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:50:29 2023

@author: huang
"""


import numpy as np
import torch
import matplotlib.pyplot as plt
import time
#from scipy import integrate
import matplotlib as mpl
import os



os.system('python FirstTaskReplicaAndSGDWithKL.py')
os.system('python SecondTaskReplicaAndSGDWithKL.py')
os.system('python FirstTaskSGDalpha1=1.4.py')



device = torch.device('cuda')
eps = torch.exp( torch.ones(1).double().cuda() )**(-700)

def loss(q, r, q_star, beta, largealpha = True):
    k = beta
    z = torch.randn(10000,1).to(device).double()
    if not largealpha:
        m = 2*torch.rand(1,10000).to(device).double() -1
        #B = torch.tensor([1,-1]).to(device).reshape(1,1,2)
        f = -(1+m)/2*torch.log((1+m)/2)   -(1-m)/2*torch.log((1-m)/2)
        h = (q_star - q/2 + eps)*m**2 + torch.sqrt(q + eps)*z*m + r*m + k*f
        #print((h!=h).sum())
        h = h-h.max()
        D = torch.exp(h).mean(1,keepdim = True) 
        M = torch.exp(h)
        intterm = M/D
        intterm[D.reshape(-1)==0,:] = 0

        p = (( torch.sign(m)  )*intterm).mean()
        l = torch.arccos(p)/np.pi
        
    else:
        #alpha比较大时，应该是(\sqrt q_hat z + r_hat )m占主导，可以利用反函数法采样这个分布，然后作平均
        phi = torch.rand(1,10000).to(device).double()
        m = (torch.log(torch.exp( torch.sqrt(q)*z + r -400)* phi +  (1-phi)* torch.exp(-torch.sqrt(q)*z - r-400)) + 400 )/(torch.sqrt(q)*z + r)
        f = -(1+m)/2*torch.log((1+m)/2)   -(1-m)/2*torch.log((1-m)/2)
        h = (q_star - q/2 + eps)*m**2  + k*f
        #print(m.max(),m.min(), h.max())
        D = torch.exp(h).mean(1,keepdim = True) 
        M = torch.exp(h)
        intterm = M/D
        intterm[D.reshape(-1)==0,:] = 0

        p = (( torch.sign(m)  )*intterm).mean()
        l = torch.arccos(p)/np.pi
        
            
    
    
    return l,p








results =[]
results_hat = []
l_list = []
p_list = []
for i in range(24):
    a,b,c,d = torch.load( 'data/results_grid'+str((i+1)/10)+'beta100')
    results.append(a)
    results_hat.append(b)
    l_list.append(c)
    p_list.append(d)

    


results = np.array(results)
results_hat = np.array(results_hat)
p_list = torch.tensor(p_list).cpu().numpy()
l_list = torch.tensor(l_list).cpu().numpy()





device = torch.device('cuda')
eps = 1e-30
l_list = []
p_list = []
for j in range( 50 ):
    print(j, end = ';')
    l, p =[], []
    i=0
    while i < len(results):
        if i>=20:
            largealpha = True
        else:
            largealpha = False
        temp = torch.from_numpy(results_hat[i].mean(0)).cuda()
        #print(';;;')
        a,b = loss(temp[0], temp[1], temp[2], 100, largealpha = largealpha)
        #print(a)
        if a ==a :
            i+=1
            l.append(a.item())
            p.append(b.item())
    l_list.append(l)
    p_list.append(p)
l100 = np.array(l_list)
p100 = np.array(p_list)




l_GD = []
p_GD = []
for i in range(12):
    a = torch.load('data/GD_alpha='+str((i+1)/5)+'-withKL')
    l_GD.append([a[i][0] for i in range(len(a))])
    p_GD.append([a[i][1] for i in range(len(a))])

    
l_GD = np.array(l_GD)
p_GD = np.array(p_GD)
l_GD = np.arccos(p_GD)/np.pi









color = ['black', 'gray', 'black']
plt.figure(figsize = (6,5))
axis = plt.subplot(111)
x = [(i+1)/5 for i in range(l_GD.shape[0]-1)] + [2.1]
argsort =[i for i in  np.argsort(x)]
plt.plot([(i+1)/10 for i in range(len( l100.mean(0)))], l100.mean(0), marker = 'o', markerfacecolor='white', color = color[1], label = r'$\epsilon_g$', linewidth = 3)
plt.errorbar([x[argsort[i]] for i in range(len(x))], [l_GD.mean(1)[argsort[i]] for i in range(len(x))], [l_GD.std(1)[argsort[i]] for i in range(len(x))],capsize = 3,markerfacecolor='white',marker = 's',linestyle = '--', color = color[1], label = r'$\epsilon_g^{SGD}$', linewidth = 2 )

plt.plot([(i+1)/10 for i in range(len( l100.mean(0)))], p100.mean(0), marker = 'o', markerfacecolor='white', color = color[0], label = r'$p_1$', linewidth = 3)
plt.errorbar([x[argsort[i]] for i in range(len(x))], [p_GD.mean(1)[argsort[i]] for i in range(len(x))], [p_GD.std(1)[argsort[i]] for i in range(len(x))],capsize = 3,markerfacecolor='white',marker = 's',linestyle = '--', color = color[0], label = r'$p^{SGD}_1$', linewidth = 2 )

plt.legend(fontsize = 15)
#plt.ylabel(r'$\epsilon_g$', fontsize = 30)

plt.xlabel(r'$\alpha$', fontsize = 30)
plt.tick_params(which = 'major', direction = 'in', width = 4, length = 8)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
plt.savefig('fig/fig-A1-a.pdf', bbox_inches = 'tight')












def Loss(q_1, r_1, q_1star, q_2, r_2, q_2star, k, T, p_0 = 0, r_0 = 0):
    p_0 = 0
    z = torch.randn(500,1,1,1,1).to(device).double()
    x = torch.randn(1,500,1,1,1).to(device).double()
    m_1 = 2*torch.rand(1,1,500,1,1).to(device).double() -1
    m = 2*torch.rand(1,1,1,500,1).to(device).double() - 1
    k1 = 100
    partial_p1_r, partial_p2_r = 0, 0

    B_2 = torch.tensor([1,-1]).reshape(1,1,1,1,2).to(device).double()
    f1 = -(1+m_1)/2*torch.log((1+m_1)/2) - (1-m_1)/2*torch.log((1-m_1)/2)
    g = (q_1star - q_1/2)*m_1**2 + torch.sqrt(q_1 + eps)*z*m_1 + r_1*m_1 + k1*f1
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






_,results_hat,_,_ = torch.load('data/results_grid'+str(20/10))
q_1_hat, r_1_hat, q_1star_hat = torch.tensor(results_hat).cuda().mean(0).double()
r_0_list = [-0.5,0,0.5]
alpha_list = [5*(i+1)/10 for i in range(9)]
results2 = []
for r_0 in r_0_list:
    result = []
    for alpha_2 in alpha_list:
        result.append( torch.load( 'data/second-task-'+'alpha-'+str(alpha_2) + 'r_0-'+(str(r_0))+'-beta1=100-task1withKL' )   )  
        #print(result[-1])
    results2.append(result)
results2 = torch.tensor(results2).cuda()



q_2 =  results2[:,:,:,0].mean(2)
r_2 =  results2[:,:,:,1].mean(2)
q_2star =  results2[:,:,:,2].mean(2)
q_2_hat = results2[:,:,:,5].mean(2)
r_2_hat = results2[:,:,:,6].mean(2)
q_2star_hat = results2[:,:,:,7].mean(2)

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
            a,b,c,d = Loss(q_1_hat, r_1_hat, q_1star_hat, q_2_hat[i,j], r_2_hat[i,j], q_2star_hat[i,j], 100, 0, p_0 = 0, r_0 = r_0_list[i])
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
alpha_list = [(i+1)/2 for i in range(9)]
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
        a = torch.load( 'data/GD_alpha_second='+str(alpha_1) +','+str(alpha_2)+'r_0='+str(r_0)+'-task1KL')
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
    
    plt.plot([5*(i+1)/10 for i in range(len(l_2[0]))],l_2[i].cpu(),'-',marker = marker[i],markersize = 8, markerfacecolor = "None", label = label_r[i], color = color[i])
   
    plt.errorbar([(i+1)/2 for i in range(9)], l_2_GD[i].mean(1), l_2_GD[i].std(1),linestyle = '--', 
                     capsize = 3,markersize = 8,markerfacecolor = "None",marker = marker[i], color = color1[i])
    #plt.plot([(i+1)/2 for i in range(10)], l_1[i].cpu(), label = label_r[i])
plt.legend(fontsize = 15)
plt.xlabel(r'$\alpha_2$', fontsize = 30)
plt.ylabel(r'$\epsilon_g^2$', fontsize = 30)
#plt.title(r'$\gamma = 1$', fontsize = 30)
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
plt.savefig('fig/fig-A1-b.pdf', bbox_inches = 'tight')




label = [r'$q_2$', r'$r_2$', r'$q_{2,\star}$']
label_r = [r'$r_0=-0.5$', r'$r_0=0$', r'$r_0=0.5$']
color = ['r', 'b', 'black']
color1 = ['salmon', 'royalblue', 'dimgray']
plt.figure(figsize = (6,5))
axis = plt.subplot(111)
for i in range(3):
    plt.plot([5*(i+1)/10 for i in range(len(l_2[0]))],l_1[i].cpu(),'-',marker = marker[i],markersize = 8, markerfacecolor = "None", label = label_r[i], color = color[i])
   
    plt.errorbar([(i+1)/2 for i in range(9)], l_1_GD[i].mean(1), l_1_GD[i].std(1),linestyle = '--', 
                     capsize = 3,markersize = 8,markerfacecolor = "None",marker = marker[i], color = color1[i])
    #plt.plot([(i+1)/2 for i in range(10)], l_1[i].cpu(), label = label_r[i])
plt.legend(fontsize = 15)
plt.xlabel(r'$\alpha_2$', fontsize = 30)
plt.ylabel(r'$\epsilon_g^1$', fontsize = 30)
#plt.title(r'$\gamma = 1$', fontsize = 30)
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
plt.savefig('fig/fig-A1-c.pdf', bbox_inches = 'tight')










loss = []
for alpha_2 in [(i+1)/2 for i in range(9)]:
    
    a = (torch.load('data/GD_alpha_second='+str(alpha_1) +','+str(alpha_2)+'r_0='+str(0)+'-alpha1=1.4'))
    loss.append(torch.tensor(a).cpu().numpy())
loss = np.array(loss)




plt.figure(figsize = (6,5))
axis = plt.subplot(111)
plt.errorbar([(i+1)/2 for i in range(9)], loss.mean(1)[:,1], loss.std(1)[:,1],linestyle = '--', 
                     capsize = 3,markersize = 8,markerfacecolor = "None",marker = 'o', color = 'black', label = r'$\epsilon_g^2$')
plt.errorbar([(i+1)/2 for i in range(9)], loss.mean(1)[:,0], loss.std(1)[:,0],linestyle = '--', 
                     capsize = 3,markersize = 8,markerfacecolor = "None",marker = 's', color = 'gray', label = r'$\epsilon_g^1$')
plt.legend(fontsize = 15)
plt.xlabel(r'$\alpha_2$', fontsize = 30)
#plt.ylabel(r'$\epsilon_g^2$', fontsize = 30)
#plt.title(r'$\gamma = 1$', fontsize = 30)
plt.tick_params(which = 'major', direction = 'in', width = 4, length = 8)
plt.tick_params(which = 'minor', direction = 'in', width = 1, length = 5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
plt.savefig('fig-A1-d.pdf', bbox_inches = 'tight')