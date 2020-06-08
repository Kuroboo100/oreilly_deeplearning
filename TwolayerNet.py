import numpy as np
import matplotlib.pyplot as plt
import layers as ly
from collections import OrderedDict

class TwoLayerNet():
    def __init__(self,input_size,hidden_size,output_size,weight_init_std):
        #parameter
        self.params={}
        self.params["W1"]=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params["b1"]=weight_init_std*np.random.randn(hidden_size)
        self.params["W2"]=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params["b2"]=weight_init_std*np.random.randn(output_size)
        
        #layer
        self.layers=OrderedDict()
        self.layers["affine1"]=ly.Affine(self.params["W1"],self.params["b1"])
        self.layers["Sigmoid"]=ly.SigmoidLayer()
        self.layers["affine2"]=ly.Affine(self.params["W2"],self.params["b2"])
        self.lastlayer=ly.SoftmaxwithLoss()

#誤差逆伝播法による実装
    def predict(self,x):
        for layer in self.layers.values():
            y=layer.forword(x)
            x=y
        return y
    
    def loss_cal(self,y,t):
        return self.lastlayer.forword(y,t)
    
    def accuracy(self,y,t):
        batch_size=y.shape[0]
        ac=0
        for n in range(batch_size):
            if np.argmax(y[n])==np.argmax(t[n]):
                ac+=1
        return ac/batch_size*100

    def gradient(self,dout):
        dout=self.lastlayer.backword(dout)
        l=list(self.layers.values())
        l.reverse()
        y=dout
        for layer in l:
            x=layer.backword(y)
            y=x
        
        grads={}
        grads["W1"]=self.layers["affine1"].dW
        grads["b1"]=self.layers["affine1"].db
        grads["W2"]=self.layers["affine2"].dW
        grads["b2"]=self.layers["affine2"].db
        return grads

#勾配法による実装
    def total_differential(self,f,W):
        h=10e-4
        grad=np.zeros_like(W)
        for i,cell in np.ndenumerate(W):
            tmp=W[i]
            W[i]=tmp+h
            fp=f(W)
            W[i]=tmp-h
            fm=f(W)
            W[i]=tmp
            grad[i]=(fp-fm)/(2*h)
        return grad
    
    def num_grad(self,y,t_batch):
        grads={}
        for key in self.params:
            W=self.params[key]
            f=lambda W:self.loss_cal(y,t_batch)
            grads[key]=self.total_differential(f,W)
        return grads