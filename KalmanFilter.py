
# coding: utf-8

# In[43]:


import copy as cp
import math as mt
import numpy as np
import pandas as pd

def rmat(nr,nc): return(np.matrix(np.random.rand(nr,nc)))

def 上三角(s):
    from scipy.sparse import triu
    return np.matrix(triu(rmat(s,s)).toarray())
    

class parameters:
    def __init__(self,size):
        self.size = size
        self.mu = np.matrix(np.zeros((size,1)))
        self.F = np.matrix(np.zeros((size,size)))
        self.Q = np.matrix(np.zeros((size,size)))
        self.R = np.matrix(np.zeros((1,1)))
        self.beta00 = np.matrix(np.zeros((size,1)))
        self.P00 = np.matrix(np.zeros((size,size)))
        self.parnames = list(self.__dict__.keys())
        self.parnames.remove('size')
        self.parnames.sort()
        
    def pack(self):
        p = np.matrix(np.zeros((0,1)))
        for aname in parnames:
            mtrx = getattr(ibm_model.parameters,aname)
            newmtrx = np.reshape(mtrx,(-1,1),order='F')
            p = np.concatenate((p,newmtrx),axis=0)
        return(p)
    
    def unpack(self,p):
        pos = 0
        for aname in self.parnames:
            mtrx = getattr(self,aname)
            nr,nc = np.shape(mtrx)
            newpos = pos + nr*nc
            subp = np.matrix(p[pos:newpos])
            mtrx = np.reshape(subp,(nr,nc),order='F')
            setattr(self,aname,mtrx)
            pos = cp.copy(newpos)
            
        
    def randomize(self):
        s = self.size
        self.mu = rmat(s,1)
        self.F = rmat(s,s)
        Q = 上三角(s)
        self.Q = Q.transpose()*Q
        self.R = rmat(1,1)
        self.beta00 = rmat(s,1)
        P = 上三角(s)
        self.P00 = P.transpose()*P
        
        
class thedata:
    def __init__(self,size):
        self.y = np.matrix(np.zeros((1,1)))
        self.x = np.matrix(np.zeros((1,size)))
        
class predictions:
    def __init__(self,size):
        self.beta = np.matrix(np.zeros((size,1)))
        self.P = np.matrix(np.zeros((size,size)))
        self.eta = np.matrix(np.zeros((1,1)))
        self.f = np.matrix(np.zeros((1,1)))
        
class updating:
    def __init__(self,size):
        self.beta = np.matrix(np.zeros((size,1)))
        self.P = np.matrix(np.zeros((size,size)))
        self.K = np.matrix(np.zeros((size,1)))
        
class an_obs:
    def __init__(self,t,fltr):
        self.t = t
        self.fltr = fltr
        size = self.fltr.size
        self.data = thedata(size)
        self.prediction = predictions(size)
        self.updating = updating(size)
    def predict(self):
        if self.t == 0: 
            b00 = self.fltr.parameters.beta00
            P00 = self.fltr.parameters.P00
        else: 
            b00 = self.fltr.obs[self.t-1].updating.beta
            P00 = self.fltr.obs[self.t-1].updating.P
        F = self.fltr.parameters.F
        x = self.data.x
        self.prediction.beta = self.fltr.parameters.mu+F*b00
        self.prediction.P = F*P00*F.transpose()+self.fltr.parameters.Q
        eta = self.data.y - x*self.prediction.beta
        f = x*self.prediction.P*x.transpose()+self.fltr.parameters.R
        self.prediction.eta = eta
        self.prediction.f = f
        self.ll = mt.log(2*mt.pi*f) +eta.transpose()*np.linalg.inv(f)*eta
    def update(self):
        P = self.prediction.P
        x = self.data.x
        K = P*x.transpose()*np.linalg.inv(self.prediction.f)
        self.updating.K = K
        self.updating.beta = self.prediction.beta + K*self.prediction.eta
        self.updating.P = P - K*x*P

class a_filter:
    def __init__(self,thefile):
        data = pd.read_csv(thefile)
        nr,nc = np.shape(data)
        nx = nc-1
        self.intercept = False
        response = input ('Please enter "NO" if you do not want an intercept:')
        if (response != "NO"):
            nx = nx+1
            self.intercept=True
        self.size = nx
        self.T = nr
        self.parameters = parameters(self.size)
        self.obs =[]
        for t in range(self.T):
            newobs = an_obs(t,self)
            x = np.matrix(data.iloc[t,1:])
            y = np.matrix(data.iloc[t,0])
            if self.intercept:
                x = np.concatenate((np.matrix(1),x),axis=1)
            newobs.data.y = y
            newobs.data.x = x
            self.obs = self.obs + [newobs]
            
    def run(self):
        self.ll = 0
        for t in range(self.T):
            self.obs[t].predict()
            self.obs[t].update()
            self.ll = self.ll + self.obs[t].ll
        self.ll = -0.5*self.ll
            
            


# In[44]:


ibm_model = a_filter('IBM.csv')
ibm_model.parameters.randomize()
p = ibm_model.parameters.pack()


# In[48]:


newparameters=parameters(4)


# In[49]:


newparameters.__dict__


# In[50]:


newparameters.unpack(p)


# In[51]:


newparameters.F


# In[52]:


ibm_model.parameters.F


# In[53]:


newparameters.Q


# In[54]:


ibm_model.parameters.Q


# In[34]:


ibm_model = a_filter('IBM.csv')


# In[35]:


ibm_model.obs[7].data.x


# In[36]:


ibm_model.parameters.randomize()


# In[37]:


ibm_model.parameters.pack()


# In[64]:


ibm_model.obs[7].prediction.f


# In[6]:


ibm_model.parameters.randomize()


# In[7]:


ibm_model.run()


# In[8]:


ibm_model.ll


# In[9]:


ibm_model.parameters.__dict__.keys()


# In[10]:


for akey in list(ibm_model.parameters.__dict__.keys()):
    print ('this is the key:'+akey)


# In[11]:


F = ibm_model.parameters.F


# In[12]:


np.shape(F)


# In[14]:


np.reshape(F,(16,1),order='F')


# In[16]:


FF = np.reshape(F,(16,1),order='F')


# In[17]:


np.reshape(FF,(4,4),order='F')


# In[18]:


np.reshape(F,(-1,1),order='F')


# In[19]:


np.matrix(np.zeros((0,1)))


# In[20]:


np.concatenate((np.matrix(np.zeros((0,1))),FF),axis=0)


# In[21]:


parnames = list(ibm_model.parameters.__dict__.keys())


# In[22]:


parnames


# In[23]:


parnames.remove('size')


# In[24]:


parnames


# In[25]:


parnames.sort()


# In[26]:


parnames


# In[ ]:


p = np.matrix(np.zeros((0,1)))
for aname in parnames:
    mtrx = ibm_model.parameters.aname
    newmtrx = np.reshape(mtrx,(-1,1),order='F')
    p = np.concatenate((p,newmtrx),axis=0)


# In[27]:


getattr(ibm_model.parameters,'F')


# In[28]:


whatevernameyouwant='F'


# In[29]:


getattr(ibm_model.parameters,whatevernameyouwant)


# In[32]:


p = np.matrix(np.zeros((0,1)))
for aname in parnames:
    mtrx = getattr(ibm_model.parameters,aname)
    newmtrx = np.reshape(mtrx,(-1,1),order='F')
    p = np.concatenate((p,newmtrx),axis=0)

