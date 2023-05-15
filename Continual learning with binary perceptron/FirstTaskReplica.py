# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:31:16 2023

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
    
    return partial_q, partial_r, partial_q_star




def partial_S(q,r,q_star, beta, largealpha = False):
    k = beta
    z = torch.randn(10000,1).to(device).double()
    if not largealpha:
        m = 2*torch.rand(1,10000).to(device).double() -1
        h = (q_star - q/2 + eps)*m**2 + torch.sqrt(q + eps)*z*m + r*m 
        h = h - h.max()
        D = torch.exp(h).mean(1,keepdim = True) + eps
        M = torch.exp(h)
        intterm = M/D
        intterm[D.reshape(-1)==0,:] = 0
    else:
        # when alpha is large, sample a proposal distribution
        phi = torch.rand(1,10000).to(device).double()
        m = (torch.log(torch.exp( torch.sqrt(q)*z + r -400)* phi +  (1-phi)* torch.exp(-torch.sqrt(q)*z - r-400)) + 400 )/(torch.sqrt(q)*z + r)
        h = (q_star - q/2 + eps)*m**2  
        D = torch.exp(h).mean(1,keepdim = True) 
        M = torch.exp(h)
        intterm = M/D
        intterm[D.reshape(-1)==0,:] = 0
        
    partial_q = ((-m**2/2 + m*z/2/(torch.sqrt(q + eps)))*intterm).mean()
    partial_r = (( m  )*intterm).mean()
    partial_q_star = (( m**2  )*intterm).mean()
    
    z = m=f=h=D=M=0#to release cuda memory
    
    
    return partial_q, partial_r, partial_q_star






def beta_exp(alpha, beta):
    beta_list = [beta]
    alpha = alpha
    if alpha>=1.5:
        largealpha = True
    else:
        largealpha = False
    results = []
    results_hat = []
    resultss = []
    eta = 0.9 # Damping coefficient
    for x in range(10):
        t1 = time.time()
        for l in range(len(beta_list)):
            beta = beta_list[l]
            q =  torch.rand(1).to(device).double()
            r = torch.rand(1).to(device).double()*q
            q_star = torch.rand(1).to(device).double()*(1-q) +q
            
            q_hat =r_hat =q_star_hat=0
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
                
                print('.', q, r, q_star , q_hat,r_hat,q_star_hat, time.time()-t1,end = ';' )
                j = 0
                k = 0
                temp1 = torch.zeros([20,3]).to(device).double()
                while(j<20):
                    temp1[j][0],temp1[j][1],temp1[j][2]  = partial_S(q_hat, r_hat, q_star_hat, beta, largealpha = largealpha)
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
                if (i == 99) or ( err <4e-6*eta**2):
                    results.append([q.item(), r.item(), q_star.item()])
                    results_hat.append([q_hat.item(), r_hat.item(), q_star_hat.item()])
                    print('alpha',alpha,'beta',beta, ';',i,';',err)
                    print(q,r,q_star,q_hat, r_hat, q_star_hat, end = ';;;')
                    #torch.save([results,results_hat], 'results_alpha=0.8')
                    break
        
        resultss.append([results, results_hat])
    torch.save([results,results_hat], 'data/results_grid'+str(alpha))
    
    
    
    
for i in range(200):
    beta_exp((i+1)/100, 20)