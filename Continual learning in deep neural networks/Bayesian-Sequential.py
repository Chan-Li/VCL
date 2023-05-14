#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cupy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numpy import random
import math
from matplotlib.pyplot import plot,savefig
import mnist_reader
train_data4f, train_label4f = mnist_reader.load_mnist('fashion', kind='train',normalize=True, one_hot=True)
test_data4f, test_label4f = mnist_reader.load_mnist('fashion', kind='t10k',normalize=True, one_hot=True)
train_data4 = train_data4f[0:64*500].T*1
test_data4 = test_data4f.T*1
train_label4 = train_label4f[0:64*500].T*1
test_label4 = test_label4f.T*1
#shapes: 784*30000// 10*30000


# In[6]:


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
        soft.append(cut)
    return np.array(soft).T
def divi_(lr0,global_step,decay_step):
    return lr0*(0.5**((int((global_step)/decay_step))))
import load
mnist=(load.load_mnist(one_hot=True))
train_data0 = mnist[0][0][0:64*500].T
train_label0 = mnist[0][1][0:64*500].T
test_data0 = mnist[1][0][:10000].T
test_label0 = mnist[1][1][:10000].T
def uni_permu(a,b,direction):
    if direction ==1:
        p = np.random.permutation(len(a.T))
        return ((a.T[p]).T), ((b.T[p]).T)
    if direction == 0:
        p = np.random.permutation(len(a))
        return ((a[p])), ((b[p]))
def mini_batch_generate(mini_batch_size,data1,label1):
    data = np.array(data1*1)
    label = np.array(label1*1)
    if (data.shape[1]%mini_batch_size == 0):
        n=data.shape[1]
    else:
        n = (int(data.shape[1]/mini_batch_size))*mini_batch_size
    data,label = uni_permu(data,label,1)
    mini_batches = np.asarray([data[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    mini_batches_labels =np.asarray([label[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    return (mini_batches),(mini_batches_labels)
def turn_2_zero(x):
    y = x*1
    z = np.ones((np.array(x).shape))
    z[y==0] = 0
    return z
def Bolzmann(theta,w,beta):
    pro = (pow(np.e,beta*w*theta)/(pow(np.e,beta*theta)+pow(np.e,-beta*theta))).reshape(theta.shape)
    return np.array(pro)
##采样出二值化的w
def sampling(theta,beta):
    theta=np.array(theta)
    w_sam=np.ones_like(theta)
    pro = Bolzmann(theta,w_sam,beta)
    ran=np.array(np.random.random(size=(theta.shape)))
    w_sam[(pro-ran)<=0] = -1
    return w_sam


# ## Updating rule: $\Delta \theta_{ij}^{-l} = -\eta\left[\mathcal{K}_{i}^{-l} \left(\frac{1}{\sqrt{N_{-l-1}}}\beta a_{j}^{-l-1}-\beta \frac{\epsilon_{i}^{-l}}{N_{-l-1} v_{i}^{-l}}\left(a_{j}^{-l-1}\right)^{2} \mu_{i j}^{-l}\right)\left(\sigma_{i j}^{-l}\right)^{2}+\beta^{2}(\sigma_{ij}^{-l})^{2}\Delta_{ij,l}^{t-1} \right]$

# ## when l =L, $\mathcal{K}_{i}^{L} = (y_i - \hat{y}_i)$

# ## when 1<l<L, $\mathcal{K}_{j}^{-l} = \frac{\partial C}{\partial z_{j}^{-l}} =\sum_{k} \frac{\partial C}{\partial z_{k}^{-l+1}} \frac{\partial z_{k}^{-l+1}}{\partial z_{j}^{-l}}=\sum_{k} \mathcal{K}_{k}^{-l+1}\left(\frac{1}{\sqrt{N_{-l}}}\mu_{kj}^{-l+1}+\frac{\epsilon_{k}^{-l+1}}{N_{-l}\sqrt{\left(v_{k}^{-l+1}\right)^{2}}}\left(\sigma_{kj}^{-l+1}\right)^{2} a_{j}^{-l}\right) f^{\prime}\left(z_{j}^{-l}\right) $

# In[9]:


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
def beta_func(ep_all,ep):
    return (10*np.tanh(0.1+1/(ep_all)*(ep+1)))
class NeuralNetwork:
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.beta =1.0
        self.sizes = sizes
        self.theta = [(np.random.normal(0,1,size=(y,x))) for x, y in zip(self.sizes[:-1], self.sizes[1:])]    
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
            zm.append(z)
            process.append(a)
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
            zm.append(z)
            process.append(a)
            epsilon.append(ep)
            var.append(v)
        if back == False:
            return process[-1]
        if back == True:
            return process,epsilon,zm,var
        
    def evaluate(self, testdata1,testdata2,activate):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        self.update_moment()
        data1,label1 = mini_batch_generate(64,testdata1*1,test_label0*1)
        data2,label2 = mini_batch_generate(64,testdata2*1,test_label4*1)
        accuracy1=[]
        accuracy2 = []
        for j in range(data1.shape[0]):
            a=self.feedforward(data1[j],activate,back=False)
            max0=np.argmax(a,axis=0)
            max1=np.argmax(label1[j],axis=0)
            accuracy1.append((np.sum((max0-max1) == 0))/(data1[j].shape[1]))
        for j in range(data2.shape[0]):
            b=self.feedforward(data2[j],activate,back=False)
            max0=np.argmax(b,axis=0)
            max1=np.argmax(label2[j],axis=0)
            accuracy2.append((np.sum((max0-max1) == 0))/(data2[j].shape[1]))
        return np.average(accuracy1),np.average(accuracy2)
    

   
       

    
    
    
    
    def sampling_evaluate(self, testdata1,testdata2,activate):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        self.update_moment()
        data1,label1 = mini_batch_generate(64,testdata1*1,test_label0*1)
        data2,label2 = mini_batch_generate(64,testdata2*1,test_label4*1)
        accuracy1=[]
        accuracy2 = []
        for j in range(data1.shape[0]):
            a=self.w_feedforward(data1[j],activate,back=False)
            max0=np.argmax(a,axis=0)
            max1=np.argmax(label1[j],axis=0)
            accuracy1.append((np.sum((max0-max1) == 0))/(data1[j].shape[1]))
        for j in range(data2.shape[0]):
            b=self.w_feedforward(data2[j],activate,back=False)
            max0=np.argmax(b,axis=0)
            max1=np.argmax(label2[j],axis=0)
            accuracy2.append((np.sum((max0-max1) == 0))/(data2[j].shape[1]))
        return np.average(accuracy2),np.average(accuracy1)
    
    
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
#             print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')
        
        
    def SGD(self,mini_batch_size,epoch1,epoch2,lr0,activate,dactivate):
        evaluation_cost, evaluation_error = [], []
        training_cost, training_accuracy = [], []
        learning_rate=[]
        acc1_=[]
        acc2_=[]
        lr_=[]
        for i in range(epoch1):
            print("begin")
            self.beta = beta_func(epoch1+epoch2,i)
            print("beta = ",self.beta)
            self.update_moment()
            train_labelt=train_label0*1#改参数
            train_datat=train_data0*1
            lr=lr0
            self.adam_update(lr,mini_batch_size,activate,dactivate,train_datat,train_labelt)
            acc1,acc2 = self.evaluate(test_data0,test_data4,activate)
            print ("Task1, epoch %s training complete" % i)
            print("the test Accuracy for task0 is:{} %".format((acc1)*100))
            print("the test Accuracy for task1 is:{} %".format((acc2)*100))
            acc1_.append(acc1)
            acc2_.append(acc2)
            lr_.append(lr)
        for i in range(epoch2):
            self.beta = beta_func(epoch1+epoch2,epoch1+i)
            print("beta = ",self.beta)
            self.update_moment()
            train_labelt=train_label4*1#改参数
            train_datat=train_data4*1
            lr = lr0
            self.adam_update(lr,mini_batch_size,activate,dactivate,train_datat,train_labelt)
            acc1,acc2= self.evaluate(test_data0,test_data4,activate)
            print ("Task2, epoch %s training complete" % i)
            print("the test Accuracy for task0 is:{} %".format((acc1)*100))
            print("the test Accuracy for task1 is:{} %".format((acc2)*100))
            acc1_.append(acc1)
            acc2_.append(acc2)
            lr_.append(lr)

        return acc1_,acc2_,lr_




# In[10]:


if __name__ == '__main__':
    acc11=[]
    acc22=[]
    weight = []
    net = NeuralNetwork([784,512,512,10])
    acc1,acc2,lr = net.SGD(64,40,40,0.005,relu,drelu) # with KL
    print("last accuracy for task1 %s " % acc2[-1])
    print("last accuracy for task2 %s " % acc1[-1])
    acc11.append(acc1*1)
    acc22.append(acc2*1)
