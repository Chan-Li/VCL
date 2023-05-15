# -*- coding: utf-8 -*-
"""
Created on Sat May 13 19:27:56 2023

@author: huang
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time


device = torch.device('cuda')
eps = 1e-50
def H(x):
    y = -1/2*torch.erf(x/np.sqrt(2)) + 1/2
    return y

def H_prime(x):
    #print(x.shape)
    return -torch.exp(-x**2/2)/np.sqrt(2*np.pi)



def partial_E(q, r, q_star, beta, p_0 = 0):
    
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
    g = (q_1star - q_1/2)*m_1**2 + torch.sqrt(q_1 + eps)*z*m_1 + r_1*m_1 
    f = -(1+m)*torch.log( (1 + m)/(1 + m_1) )/2 - (1-m)*torch.log( (1-m) / (1-m_1 ))/2
    h = (q_2star - q_2/2)*m**2 +torch.sqrt(q_2 + eps)*x*m + k*f + r_2*m*B_2 + T*m
    D = torch.exp(g).mean(2, keepdim = True)*2 + eps
    
    
    
    M1 = (torch.exp(h) * (-m**2/2 + x*m /2/ torch.sqrt(q_2 + eps))).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True)+ eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_q2 = (M2/D)
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
    for times in range(20):
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
                                        partial_S(q_1_hat, r_1_hat, q_1star_hat, q_2_hat, r_2_hat, q_2star_hat, k_hat, T_hat, p_0 = 0, r_0 = r_0)

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
    torch.save(results, 'data/second-task-'+'alpha-'+str(alpha_2) + 'r_0-'+(str(r_0)))








_,results_hat,_,_ = torch.load('data/results_grid'+str(200/100))


q_1_hat, r_1_hat, q_1star_hat = torch.tensor(results_hat).cuda().mean(0).double()
r_0_list = [-0.5,0,0.5]
alpha_list = [0.1,0.01] + [5*(i+1)/10 for i in range(10)]
for r_0 in r_0_list:
    for alpha_2 in alpha_list:
        exp(alpha_2, 20, q_1_hat, r_1_hat, q_1star_hat, r_0)









## gamma = 0.1
gamma = 0.1
def partial_S(q_1, r_1, q_1star, q_2, r_2, q_2star, k1, k, T, p_0 = 0, r_0 = 0.5):
    p_0 = 0
    z = torch.randn(500,1,1,1,1).to(device).double()
    x = torch.randn(1,500,1,1,1).to(device).double()
    m_1 = 2*torch.rand(1,1,500,1,1).to(device).double() -1
    m = 2*torch.rand(1,1,1,500,1).to(device).double() - 1
    
    partial_q2_r, partial_r2_r, partial_q2star_r, partial_k_r, partial_T_r = 0,0,0,0,0



    B_2 = torch.tensor([1,-1]).reshape(1,1,1,1,2).to(device).double()
    g = (q_1star - q_1/2)*m_1**2 + torch.sqrt(q_1 + eps)*z*m_1 + r_1*m_1 
    f = -(1+m)*torch.log( (1 + m)/(1 + m_1) )/2 - (1-m)*torch.log( (1-m) / (1-m_1 ))/2
    f*=gamma
    h = (q_2star - q_2/2)*m**2 +torch.sqrt(q_2 + eps)*x*m + k*f + r_2*m*B_2 + T*m
    D = torch.exp(g).mean(2, keepdim = True)*2 + eps
    
    
    
    M1 = (torch.exp(h) * (-m**2/2 + x*m /2/ torch.sqrt(q_2 + eps))).mean(3, keepdim = True)/ (torch.exp(h).mean(3, keepdim = True)+ eps)
    M2 = (torch.exp(g)*M1).mean(2, keepdim = True)*2
    partial_q2 = (M2/D)
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
    
    for times in range(5):
        k = -torch.rand(1).to(device).double()
        T = torch.rand(1).to(device).double()
        temp_ = 0
        q_2_hat = 0
        r_2_hat = 0
        q_2star_hat = 0
        q_2 =  0.1*torch.rand(1).to(device).double()
        r_2 = 0.1*torch.rand(1).to(device).double()*q_2
        q_2star = 0.1*torch.rand(1).to(device).double()*q_2 + q_2
        k = -torch.rand(1).to(device).double()
        T = torch.rand(1).to(device).double()
        eta = 0.9
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



            q_2_hat_temp, r_2_hat_temp, q_2star_hat_temp = temp1[:j,:].mean(0)
            

            q_2_hat_temp *= (-2*alpha_2)
            r_2_hat_temp *= alpha_2
            q_2star_hat_temp *= alpha_2
            k_hat = beta_2
            q_2_hat = q_2_hat + eta*(q_2_hat_temp-q_2_hat)
            r_2_hat = r_2_hat + eta*(r_2_hat_temp-r_2_hat)
            q_2star_hat = q_2star_hat + eta*(q_2star_hat_temp-q_2star_hat)
            T_hat = 0
            print('.....')
            j = 0
            k = 0
            n_fails = 0
            temp2 = torch.zeros(20,5).double()
            while(j<20):
                temp2[j, 0], temp2[j, 1], temp2[j, 2], temp2[j, 3], temp2[j, 4],  = \
                                        partial_S(q_1_hat, r_1_hat, q_1star_hat, q_2_hat, r_2_hat, q_2star_hat, k_hat, T_hat, p_0 = 0, r_0 = r_0)

                if not (temp2[j]!=temp2[j]).sum():
                    j+=1     
                k += 1
                if k>50:
                    print('not complete!',j,end='=j')
                    break



            q_2_temp, r_2_temp, q_2star_temp, k_temp, T_temp = temp2[:j,:].mean(0)
            q_2_temp *= (-2)
            q_2  = q_2 + eta*(q_2_temp - q_2)
            r_2 = r_2 + eta*(r_2_temp - r_2)
            q_2star = q_2star + eta*(q_2star_temp-q_2star)
            #print('ddds', q_1,  r_1, q_1star, q_2, r_2, q_2star, k, T)
            err = ((torch.tensor([q_2, r_2, q_2star]) - temp)**2).mean()
            print(err, end = ';;')
            if (i == 99) or ( err<1e-5*eta**2):
                #results.append([ q_2.item(), r_2.item(), q_2star.item(), k,T])
                #results_hat.append([ q_2_hat.item(), r_2_hat.item(), q_2star_hat.item(), ])
                print('alpha_1',alpha_2,'beta',beta_2, ';',i,';',err,';;',q_2.item(), r_2.item(), q_2star.item(),q_2_hat.item(), r_2_hat.item(), q_2star_hat.item(),r_0)
                
                
                break
        results.append([q_2.item(), r_2.item(), q_2star.item(), k,T, q_2_hat.item(), r_2_hat.item(), q_2star_hat.item(),r_0])
    torch.save(results, 'data/second-task-'+'alpha-'+str(alpha_2) + 'r_0-'+(str(r_0)) + 'smallKL'+str(gamma))
   



_,results_hat,_,_ = torch.load('data/results_grid'+str(200/100))


q_1_hat, r_1_hat, q_1star_hat = torch.tensor(results_hat).cuda().mean(0).double()
r_0_list = [-0.5,0,0.5]
alpha_list = [0.01,0.1,0.5,1,2,5]
for r_0 in r_0_list:
    for alpha_2 in alpha_list:
        t1 = time.time()
        exp(alpha_2, 20, q_1_hat, r_1_hat, q_1star_hat, r_0)
        print(time.time()-t1)