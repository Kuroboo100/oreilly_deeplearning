import numpy as np
import matplotlib.pyplot as plt

"""
入力：X 1✕3
中間層：W 3✕2
出力：2✕1
"""
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    x=list(map(lambda e:e-np.max(x),x))
    exp_x=list(map(lambda e:np.exp(e),x))
    sum_expx=np.sum(exp_x)
    return exp_x/sum_expx

def loss_cal(x,W,t):
    y=network(x,W)
    print(y)
    return -np.sum(t*np.log(y))

def network(x,W):
    y=np.dot(x,W)
    z=sigmoid(y)
    return softmax(z)

def gradient(f,x,W,t):
    h=10e-4
    W=np.ravel(W)
    grad=np.zeros_like(W)
    for idx in range(W.size):
        tmp=W[idx]
        W[idx]=tmp+h
        W=W.reshape(3,2)
        fx_p=f(x,W,t)

        W=np.ravel(W)
        W[idx]=tmp-h
        W=W.reshape(3,2)
        fx_m=f(x,W,t)
        W=np.ravel(W)
        W[idx]=tmp

        grad[idx]=((fx_p-fx_m)/(2*h))
    grad=grad.reshape(3,2)
    return grad

def graph(y):
    plt.plot(y)
    plt.show()
    return

def main():
    #初期値、設定値
    init_x=(5,3,1)
    init_W=np.random.randn(3,2)
    lr=0.5
    epoch=10
    x=init_x
    W=init_W
    t=[1,0]

    loss=[]
    for _ in range(epoch):
        loss.append(loss_cal(x,W,t))
        W-=lr*np.array(gradient(loss_cal,x,W,t))
    graph(loss)
    return

if __name__=="__main__":
    main()


