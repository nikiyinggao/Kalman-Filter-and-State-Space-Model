
# coding: utf-8

# In[6]:


# Adjust the packing and unpacking to generate positive definite symmetric covariance matrices.


# In[53]:


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
        self.halfQ = np.matrix(np.zeros((size,size)))
        self.Q = np.matrix(np.zeros((size,size)))
        self.R = np.matrix(np.zeros((1,1)))
        self.beta00 = np.matrix(np.zeros((size,1)))
        self.halfP = np.matrix(np.zeros((size,size)))
        self.P00 = np.matrix(np.zeros((size,size)))
        self.parnames = list(self.__dict__.keys())
        self.parnames.remove('size')
        self.parnames.sort()
    def pack(self):
        p = np.matrix(np.zeros((0,1)))
        for aname in self.parnames:
            mtrx = getattr(self,aname)
            if aname != 'Q' and aname != 'P':
                if aname != 'halfQ' and aname != 'halfP':
                    p = np.concatenate((p,np.reshape(mtrx,(-1,1),order='F')),axis=0)
                else:
                    for i in range(self.size):
                        p = np.concatenate((p,mtrx[0:i+1,i]),axis=0)
        return(np.array(p))
    def unpack(self,p):
        pos = 0
        for aname in self.parnames:
            if aname != 'Q' and aname != 'P':
                mtrx = getattr(self,aname)
                nr,nc = np.shape(mtrx)
                if aname != 'halfQ' and aname != 'halfP':
                    newpos = pos + nr*nc
                    subp = np.matrix(p[pos:newpos])
                    mtrx = np.reshape(subp,(nr,nc),order='F')

                else:
                    newpos = pos + int(nr*(nr+1)/2)
                    subp = np.matrix(p[pos:newpos])
                    mtrx = np.matrix(np.zeros((nr,nc)))
                    currentp = 0
                    for i in range(self.size):
                        newp = currentp+i+1
                        mtrx[0:i+1,i] = subp[currentp:newp]
                        currentp = newp
                setattr(self,aname,mtrx)
                pos = cp.copy(newpos)
        self. Q = self.halfQ.transpose()*self.halfQ
        self.P00 = self.halfP.transpose()*self.halfP
        
    def randomize(self):
        s = self.size
        self.mu = rmat(s,1)
        self.F = rmat(s,s)
        self.halfQ = 上三角(s)
        self.Q = self.halfQ.transpose()*self.halfQ
        self.R = rmat(1,1)
        self.beta00 = rmat(s,1)
        self.halfP = 上三角(s)
        self.P00 = self.halfP.transpose()*self.halfP
             
class thedata:
    def __init__(self,size):
        self.y = np.matrix(np.zeros((1,1)))
        self.x = np.matrix(np.zeros((1,size)))

class predictions:
    def __init__(self,size):
        self.beta = np.matrix(np.zeros((size,1)))
        self.P = np.matrix(np.zeros((size,size)))
        self.eta = np.matrix(np.zeros((1,1)))
        self.f =  np.matrix(np.zeros((1,1)))
        
class updating:
    def __init__(self,size):
        self.beta = np.matrix(np.zeros((size,1)))
        self.P = np.matrix(np.zeros((size,size)))
        self.K = np.matrix(np.zeros((size,1)))

class a_filter:
    def __init__(self,thefile):
        data = pd.read_csv(thefile)
        nr,nc = np.shape(data)
        nx = nc-1
        self.intercept=False
        response = input('Please enter "NO" if you do not want an intercept:')
        if (response != "NO"):
            nx = nx+1
            self.intercept=True
        self.size = nx
        self.T = nr
        self.parameters = parameters(self.size)
        self.obs=[]
        for t in range(self.T):
            newobs = an_obs(t,self)
            y = np.matrix(data.iloc[t,0])
            x = np.matrix(data.iloc[t,1:])
            if self.intercept: x = np.concatenate((np.matrix(1),x),axis=1)
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
        self.ll = mt.log(2*mt.pi*f) + eta.transpose()*np.linalg.inv(f)*eta
    def update(self):
        P = self.prediction.P
        x = self.data.x
        K = P*x.transpose()*np.linalg.inv(self.prediction.f)
        self.updating.K = K
        self.updating.beta = self.prediction.beta + K*self.prediction.eta
        self.updating.P = P - K*x*P
        


# In[54]:


ibm_model = a_filter('IBM.csv')
ibm_model.parameters.randomize()
p = ibm_model.parameters.pack()


# In[55]:


p


# In[56]:


newparameters = parameters(4)


# In[57]:


newparameters.__dict__


# In[58]:


newparameters.unpack(p)


# In[59]:


newparameters.F


# In[60]:


ibm_model.parameters.F


# In[61]:


newparameters.Q


# In[62]:


ibm_model.parameters.Q


# In[51]:





# In[49]:


F


# In[72]:


np.array(F)


# In[7]:


from scipy.optimizxe import minimize
minimize(fun=self.objective,
        x0=p0,
        method='SLSQP',
        method='COBYLA',
        constraints=cons,
        tol=0.01,
        options=opt
        ) 


# In[8]:


ssj = 上三角(5)


# In[9]:


ssj


# In[11]:


p = np.matrix(np.zeros((0,1)))
for i in range(5):
    p = np.concatenate((p,ssj[0:i+1,i]),axis=0)


# In[12]:


p


# In[63]:


# add the objective function and the constraints


# In[87]:


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
        self.halfQ = np.matrix(np.zeros((size,size)))
        self.Q = np.matrix(np.zeros((size,size)))
        self.R = np.matrix(np.zeros((1,1)))
        self.beta00 = np.matrix(np.zeros((size,1)))
        self.halfP = np.matrix(np.zeros((size,size)))
        self.P00 = np.matrix(np.zeros((size,size)))
        self.parnames = list(self.__dict__.keys())
        self.parnames.remove('size')
        self.parnames.sort()
    def pack(self):
        p = np.matrix(np.zeros((0,1)))
        for aname in self.parnames:
            mtrx = getattr(self,aname)
            if aname != 'Q' and aname != 'P':
                if aname != 'halfQ' and aname != 'halfP':
                    p = np.concatenate((p,np.reshape(mtrx,(-1,1),order='F')),axis=0)
                else:
                    for i in range(self.size):
                        p = np.concatenate((p,mtrx[0:i+1,i]),axis=0)
        return(np.array(p))
    def unpack(self,p):
        pos = 0
        for aname in self.parnames:
            if aname != 'Q' and aname != 'P':
                mtrx = getattr(self,aname)
                nr,nc = np.shape(mtrx)
                if aname != 'halfQ' and aname != 'halfP':
                    newpos = pos + nr*nc
                    subp = np.matrix(p[pos:newpos])
                    mtrx = np.reshape(subp,(nr,nc),order='F')

                else:
                    newpos = pos + int(nr*(nr+1)/2)
                    subp = np.matrix(p[pos:newpos])
                    mtrx = np.matrix(np.zeros((nr,nc)))
                    currentp = 0
                    for i in range(self.size):
                        newp = currentp+i+1
                        mtrx[0:i+1,i] = subp[currentp:newp]
                        currentp = newp
                setattr(self,aname,mtrx)
                pos = cp.copy(newpos)
        self. Q = self.halfQ.transpose()*self.halfQ
        self.P00 = self.halfP.transpose()*self.halfP
        
    def randomize(self):
        s = self.size
        self.mu = rmat(s,1)
        self.F = np.diag(np.random.rand(self.size))
        self.halfQ = 上三角(s)
        self.Q = self.halfQ.transpose()*self.halfQ
        self.R = rmat(1,1)
        self.beta00 = rmat(s,1)
        self.halfP = 上三角(s)
        self.P00 = self.halfP.transpose()*self.halfP
             
class thedata:
    def __init__(self,size):
        self.y = np.matrix(np.zeros((1,1)))
        self.x = np.matrix(np.zeros((1,size)))

class predictions:
    def __init__(self,size):
        self.beta = np.matrix(np.zeros((size,1)))
        self.P = np.matrix(np.zeros((size,size)))
        self.eta = np.matrix(np.zeros((1,1)))
        self.f =  np.matrix(np.zeros((1,1)))
        
class updating:
    def __init__(self,size):
        self.beta = np.matrix(np.zeros((size,1)))
        self.P = np.matrix(np.zeros((size,size)))
        self.K = np.matrix(np.zeros((size,1)))

class a_filter:
    def __init__(self,thefile):
        data = pd.read_csv(thefile)
        nr,nc = np.shape(data)
        nx = nc-1
        self.intercept=False
        response = input('Please enter "NO" if you do not want an intercept:')
        if (response != "NO"):
            nx = nx+1
            self.intercept=True
        self.size = nx
        self.T = nr
        self.parameters = parameters(self.size)
        self.obs=[]
        for t in range(self.T):
            newobs = an_obs(t,self)
            y = np.matrix(data.iloc[t,0])
            x = np.matrix(data.iloc[t,1:])
            if self.intercept: x = np.concatenate((np.matrix(1),x),axis=1)
            newobs.data.y = y
            newobs.data.x = x
            self.obs = self.obs + [newobs]
            
    def objective(self,p):
        self.parameters.unpack(p)
        self.run()
        print(self.ll)
        return(-self.ll)
    
    def run(self):
        self.ll = 0
        for t in range(self.T):
            self.obs[t].predict()
            self.obs[t].update()
            self.ll = self.ll + self.obs[t].ll
        self.ll = -0.5*self.ll
        
    def F_constraint(self,p):
        tmp = parameters(self.size)
        tmp.unpack(p)
        #bigmat = np.eye(self.size*self.size)-np.kron(tmp.F,tmp.F)
        #eigen_values = np.linalg.eig(bigmat)[0]
        #return(sum(eigen_values[eigen_values<=0]-0.001))
        return(sum(eigen_values[np.linalg.eig(np.eye(self.size**2)-np.kron(tmp.F,tmp.F))[0]<=0]-0.001))
    def estimate(self):
        self.parameters.randomize()
        p0 = self.parameters.pack()
        opt = []
        cons = {'type':'ineq','fun':F_constraint}
        themin = minimize(fun=self.objective,
        x0=p0,
        method='SLSQP',
        #method='COBYLA',
        constraints=cons,
        tol=0.01,
        options=opt
        )
        self.optimum = themin
    
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
        self.ll = mt.log(2*mt.pi*f) + eta.transpose()*np.linalg.inv(f)*eta
    def update(self):
        P = self.prediction.P
        x = self.data.x
        K = P*x.transpose()*np.linalg.inv(self.prediction.f)
        self.updating.K = K
        self.updating.beta = self.prediction.beta + K*self.prediction.eta
        self.updating.P = P - K*x*P
        


# In[88]:


ibm = a_filter('IBM.csv')


# In[89]:


ibm.estimate()


# In[66]:


np.matrix(np.eye(5))


# In[67]:


F = ibm_model.parameters.F


# In[68]:


F


# In[69]:


np.kron(F,F)


# In[70]:


np.linalg.eig(F)


# In[77]:


bigmat = np.eye(16)-np.kron(F,F)
eigen_values = np.linalg.eig(bigmat)[0]
badsum = 0
for a in eigen_values:
    if a < 0: badsum = badsum+a
    elif a ==0: badsum = badsum-0.001
print(badsum)


# In[78]:


eigen_values


# In[79]:


eigen_values <=0


# In[80]:


eigen_values[eigen_values <= 0]


# In[81]:


bigmat = np.eye(16)-np.kron(F,F)
eigen_values = np.linalg.eig(bigmat)[0]
sum(eigen_values[eigen_values<=0]-0.001)


# In[82]:


sum(eigen_values[np.linalg.eig(np.eye(16)-np.kron(F,F))[0]<=0]-0.001)


# In[84]:


np.diag(np.random.rand(4))

