import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def nw(x):
    #first layer
    w1=np.array([[0.1,0.3,0.6],[0.9,0.8,0.3],[0.2,0.8,0.7]])
    b1=np.array([0.5,0.2,0.3])
    #second layer
    w2=np.array([[0.1,0.3,0.6],[0.9,0.8,0.3],[0.2,0.8,0.7]])
    b2=np.array([0.7,0.2,0.3])
    #third layer
    w3=np.array([[0.1,0.2,0.6],[0.9,0.1,0.3],[0.1,0.9,0.7]])
    b3=np.array([0.2,0.2,0.3])

    y1=np.dot(x,w1)+b1
    sigout1=list(map(lambda x:sigmoid(x),y1))

    y2=np.dot(sigout1,w2)+b2
    sigout2=list(map(lambda x:sigmoid(x),y2))

    y3=np.dot(sigout2,w3)+b3

    return y3

def softmax(x):
    x=list(map(lambda e:e-np.max(x),x))
    exp_x=list(map(lambda e:np.exp(e),x))
    sum_expx=np.sum(exp_x)

    return exp_x/sum_expx
    
def main():
    #input
    x=[1,2,3]
    return softmax(nw(x))

if __name__=="__main__":
    print(main())