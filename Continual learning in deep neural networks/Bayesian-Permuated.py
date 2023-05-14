#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cupy as np
import matplotlib.pyplot as plt
from cupy import log as ln
import random
import math
from matplotlib.pyplot import plot,savefig
import time



def relu(y):
    tmp = y.copy()
    tmp[tmp < 0] = 0
    return tmp
def drelu(x):
    tmp = x.copy()
    tmp[tmp >= 0] = 1
    tmp[tmp < 0] = 0
    return tmp
def softmax(x):
    max1=np.max(x)
    return (np.exp(x-max1))/(np.sum(np.exp(x-max1)))
def softmax_more(x):
    soft=[]
    for i in range(x.shape[1]):
        cut=softmax(x[:,i])
        soft.append(cut*1)
    return np.array(soft).T
def divi_(lr0,global_step,decay_step):
    return lr0*(0.5**((int((global_step)/decay_step))))
import load
mnist=(load.load_mnist(one_hot=True))
train_data0 = np.array(mnist[0][0][0:64*500].T)
train_label0 = np.array(mnist[0][1][0:64*500].T)
test_data0 = np.array(mnist[1][0][:10000].T)
test_label0 = np.array(mnist[1][1][:10000].T)
def uni_permu(a,b,direction):
    if direction ==1:
        p = np.random.permutation(len(a.T))
        return np.array((a.T[p]).T), np.array((b.T[p]).T)
    if direction == 0:
        p = np.random.permutation(len(a))
        return np.array((a[p])), np.array((b[p]))
def mini_batch_generate(mini_batch_size,data1,label1):
    data = np.array(data1*1)
    label = np.array(label1*1)
    if (data.shape[1]%mini_batch_size == 0):
        n=data.shape[1]
    else:
        n = (int(data.shape[1]/mini_batch_size))*mini_batch_size
    data,label = uni_permu(data,label,1)
    mini_batches = np.array([data[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    mini_batches_labels =np.array([label[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    return (mini_batches),(mini_batches_labels)
def permuated_MNIST(dataset1,dataset2):
    data_x1 = (dataset1*1)
    data_x2 = (dataset2*1)
    data_xx1,data_xx2 = uni_permu(data_x1*1, data_x2*1,0)
    return (data_xx1*1),(data_xx2*1)
def Permuated_Data(dataset1,dataset2,label1,label2,task):
    datatr_all= np.zeros((task,784,64*500))
    datate_all = np.zeros((task,784,10000))
    dataset_x1 = dataset1*1
    dataset_x2 = dataset2*1
    for i in range(task):
        data_x,data_y = permuated_MNIST(dataset_x1,dataset_x2)
        datatr_all[i] = (data_x*1)
        datate_all[i] = (data_y*1)
    return np.array(datatr_all),np.array(datate_all),np.array(label1),np.array(label2)
# def Bolzmann(theta,w,beta):
#     pro = (pow(cp.e,beta*w*theta)/(pow(cp.e,beta*theta)+pow(cp.e,-beta*theta))).reshape(theta.shape)
#     return cp.array(pro)
# ##采样出二值化的w
# def sampling(theta,beta):
#     theta=cp.array(theta)
#     w_sam=cp.ones_like(theta)
#     pro = Bolzmann(theta,w_sam,beta)
#     ran=cp.array(cp.random.random(size=(theta.shape)))
#     w_sam[(pro-ran)<=0] = -1
#     return w_sam


# In[3]:


def turn_2_zero(x):
    y = x*1
    z = np.ones((np.array(x).shape))
    z[y==0] = 0
    return z
class Adam:
    def __init__(self,size):
        self.si=size
        self.lr=0.01
        self.beta1=0.9
        self.beta2=0.99
        self.epislon=1e-8
        self.m=[(np.zeros((y,x))) for x,y in zip(self.si[:-1],self.si[1:])]
        self.s=[(np.zeros((y,x))) for x,y in zip(self.si[:-1],self.si[1:])]
        self.t=0
    
    def New_theta(self,theta,gradient,eta):
        self.t += 1
        self.lr = eta*1
        self.decay=0
        g=gradient*1
        delta = [(np.zeros((y,x))) for x,y in zip(self.si[:-1],self.si[1:])]
        theta2 = [(np.zeros((y,x))) for x,y in zip(self.si[:-1],self.si[1:])]
        for l in range(len(gradient)):
            self.m[l] = self.beta1*self.m[l] + (1-self.beta1)*g[l]
            self.s[l] = self.beta2*self.s[l] + (1-self.beta2)*(g[l]*g[l])
            self.mhat = self.m[l]/(1-self.beta1**self.t)
            self.shat = self.s[l]/(1-self.beta2**self.t)
            theta2[l] = theta[l]-self.lr*((self.mhat/(pow(self.shat,0.5)+self.epislon))+self.decay*theta[l])
            delta[l] = theta2[l] - theta[l]
        return theta2*1,delta*1

class NeuralNetwork:
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.beta = 1.0
        self.sizes = sizes
        self.theta = [(np.random.normal(0.0,1.0,size=(y,x))) for x, y in zip(self.sizes[:-1], self.sizes[1:])]    
        self.mu = [np.tanh(self.beta*theta_s) for theta_s in self.theta]
        self.sigmas = [(1-(np.tanh(self.beta*theta_s))**2) for theta_s in self.theta]
        self.KL =  [(np.zeros((y,x))) for x, y in zip(self.sizes[:-1], self.sizes[1:])]    
        self.Adam_theta = Adam(self.sizes)
     

    def update_moment(self):
        self.mu = [np.tanh(self.beta*theta_s) for theta_s in self.theta]
        self.sigmas = [(1-(np.tanh(self.beta*theta_s))**2) for theta_s in self.theta]
    def w_feedforward(self,a,activate,back=False):
        flag=0
        zm=[]
        process=[a]
        self.update_moment()
        for theta_s in self.theta:
            ws = np.array(sampling(theta_s,self.beta))
            flag=flag+1
            z=(np.dot(ws,a))*(1/np.sqrt(ws.shape[1]))
            if (flag<(self.num_layers-1)):
                a = activate(z)
            if (flag>=(self.num_layers-1)):
                a = softmax_more(z)
            zm.append(z*1)
            process.append(a*1)
        if back == False:
            return process[-1]
        if back == True:
            return process,zm

        
        
        
        
        
    
    def feedforward(self,a,activate,back=False):
        #x为输入的图片，尺寸为784*mini_batch_size
        epsilon=[]
        flag=0
        zm=[]
        var=[]
        process=[a]
        for theta_s,mu_s,sigma in zip(self.theta,self.mu,self.sigmas):
            self.update_moment()
            medicine=pow(10,-30)
            flag=flag+1
            ep=np.random.normal(0,1,(theta_s.shape[0],a.shape[1]))
            mea=(np.dot(mu_s,a))*(1/(np.sqrt(theta_s.shape[1])))
            v=np.sqrt((1/theta_s.shape[1])*np.dot(sigma,a*a)+medicine)
            z=(mea+v*ep)
            if (flag<(self.num_layers-1)):
                a = activate(z)
            if (flag>=(self.num_layers-1)):
                a = softmax_more(z)
            zm.append(z*1)
            process.append(a*1)
            epsilon.append(ep*1)
            var.append(v*1)
        if back == False:
            return process[-1]
        if back == True:
            return process,epsilon,zm,var
        
    def evaluate(self, testdata,activate):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        accuracy_all = []
        for i in range(testdata.shape[0]):
            data1,label1 = mini_batch_generate(64,testdata[i]*1,test_label0*1)
            accuracy=[]
            for j in range(data1.shape[0]):
                a=self.feedforward(data1[j],activate,back=False)
                max0=np.argmax(a,axis=0)
                max1=np.argmax(label1[j],axis=0)
                accuracy.append((np.sum((max0-max1) == 0))/(data1[j].shape[1])*1)
            accuracy_all.append(np.average(accuracy)*1)
        return accuracy_all
    

   
       

    
    
    
    
    def sampling_evaluate(self, testdata,activate):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        self.update_moment()
        accuracy_all = np.zeros((5,1))
        for i in range(testdata.shape[0]):
            data1,label1 = mini_batch_generate(64,testdata[i]*1,test_label0*1)
            accuracy=0
            for j in range(data1.shape[0]):
                a=self.w_feedforward(data1[j],activate,back=False)
                max0=np.argmax(a,axis=0)
                max1=np.argmax(label1[j],axis=0)
                accuracy+=((np.sum((max0-max1) == 0))/(data1[j].shape[1]))
            accuracy_all[i] = (accuracy/data1.shape[0])
        return np.asarray(accuracy_all)
    
    
    def backprop(self,x,y,activate,dactivate,back=True):
        medicine=pow(10,-30)
        #x:输入：784*batch_size
        #y:输入标签：10*batch_size
        tri=[]
        self.update_moment()
        out,epsi,zm,va=self.feedforward(x,activate,back=True)
        var = [(vas+medicine) for vas in va]
        nabla_theta = [np.zeros(theta_s.shape) for theta_s in self.theta]
        for l in range(1, (self.num_layers)):
            self.update_moment()
            if l==1:
                tri_=(out[-1]-y)
                tri.append(tri_*1)
            else:
                tri_=((1/(np.sqrt(self.sizes[-l])))*np.dot(self.mu[-l+1].T, tri[-1])*dactivate(zm[-l]))                 + ((1/((self.sizes[-l])))*np.dot((self.sigmas[-l+1].T),(turn_2_zero(va[-l+1])*epsi[-l+1]*tri[-1]/var[-l+1]))*out[-l]*dactivate(zm[-l]))
                tri.append(tri_*1)
            nabla_theta[-l] = self.KL[-l]+((((1/np.sqrt(self.sizes[-l-1]))*np.dot(tri_,out[-l-1].T)*self.beta)            -(self.beta*(1/(self.sizes[-l-1]))*np.dot(turn_2_zero(va[-l])*epsi[-l]*tri_/var[-l],((out[-l-1].T)**2))*self.mu[-l]))*(self.sigmas[-l]))
        return nabla_theta
    
    def adam_update(self,lr,mini_batch_size,activate,dactivate,train_data_x,train_label_x):
        self.update_moment()
        data_x=train_data_x*1
        label_x=train_label_x*1
        data,label = mini_batch_generate(mini_batch_size,data_x,label_x)
        for j in range(data.shape[0]):
            self.update_moment()
            delta_nabla_theta = self.backprop(data[j],label[j],activate,dactivate,back=True)
            self.theta,delta = self.Adam_theta.New_theta(self.theta,delta_nabla_theta,lr)
            self.update_moment()
            for l in range(self.num_layers-1):
                self.KL[l] = (self.beta**2)*(np.array(self.sigmas[l]))*(delta[l])
            print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')
        

    def SGD(self,mini_batch_size,epoch,task,lr0,activate,dactivate):
        learning_rate=[]
        acc_all = []
        lr_=[]
        data_all=[]
        label_all=[]
        sigmas = []
        KL = []
        datatr_all,datate_all,label1,label2 = Permuated_Data(train_data0,test_data0,train_label0,test_label0,task)
        for j in range(task):
            train_datat = datatr_all[j]*1
            train_labelt = label1*1
            sig_=np.zeros((3,1))
            for i in range(epoch):
                self.beta = 10*np.tanh(0.1+(2/(task*epoch))*(j*epoch+i+1))
                print("beta is",self.beta)
                self.update_moment()
                lr = lr0
                sigmas.append(self.sigmas[0]*1)
                KL.append(self.KL[0]*1)
#                 sig = [np.asnumpy(sigm) for sigm in self.sigmas]
#                 for l in range(3):
#                     sig_[l] = np.average(self.sigmas[l])
#                 sigmas.append((sig_)*1)
                self.adam_update(lr,mini_batch_size,activate,dactivate,train_datat,train_labelt)
                acc1 = self.evaluate(datate_all,activate)
                acc_all.append(acc1*1)
                print(acc1)
        return (acc_all),sigmas,KL


# In[4]:


acc_a = np.zeros((5,100,5))
sigma_=np.zeros((25,512,784))
KLL2 = np.zeros((25,512,784))
if __name__ == '__main__':
    for i in range(1):
        net=NeuralNetwork([784,512,512,10])
        print ("Trying to compute %s" % i)
        acc1_all,sigma,KLL = net.SGD(64,5,5,0.01,relu,drelu)
        for j in range(25):
            acc_a[i][j] = np.asarray(acc1_all[j]*1)
            sigma_[j] = np.asarray(sigma[j]*1)
            KLL2[j] = np.asarray(KLL[j]*1)
