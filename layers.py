import numpy as np

class AddLayer:
    def __init__(self):
        pass
    def forword(self,x,y):
        return x+y
    def backword(self,dout):
        dx=dout
        dy=dout
        return dx,dy

class MultiLayer:
    #定義された時に入力x,yがあるわけではない。
    #forword関数が呼び出された時に初めてx,yが入力される。
    def __init__(self,x,y):
        self.x=None
        self.y=None
    def forword(self,x,y):
        self.x=x
        self.y=y
        out=x*y
        return out

    def backword(self,dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy

class SigmoidLayer:
    def __init__(self):
        self.out=None
    def forword(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out
    def backword(self,dout):
        return dout*self.out*(1-self.out)

class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None
    def forword(self,x):
        self.x=x
        return np.dot(x,self.W)+self.b
    def backword(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)
        return dx

class SoftmaxwithLoss:
    def __init__(self):
        self.y=None
        self.t=None
    def forword(self,y,t):
        self.y=y
        self.t=t
        batch_size=t.shape[0]
        c=np.max(y)
        exp_y=np.exp(y-c)
        sum_exp_y=np.sum(exp_y)
        s_out=exp_y/sum_exp_y
        return np.sum(-np.log(s_out)*t)/batch_size
    def backword(self,dout):
        batch_size=self.t.shape[0]
        return dout*(self.y-self.t)/batch_size

        

        