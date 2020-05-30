#勾配法を使って最小点を求めるプログラム
import numpy as np
import matplotlib.pyplot as plt

def function(x1,x2):
    return x1**2+x2**2

def numerical_gradient(f,x):
    x1=x[0]
    x2=x[1]
    d=10e-2
    for n in range(len(x)):
        dx1=(f(x1+d,x2)-f(x1-d,x2))/(2*d)
        dx2=(f(x1,x2+d)-f(x1,x2-d))/(2*d)
    return np.array([dx1,dx2])

def main():
    lr=0.1
    epoch=100
    point=[3,4]
    loss=[function(point[0],point[1])]
    for n in range(epoch):
        point-=lr*numerical_gradient(function,point)
        loss.append(function(point[0],point[1]))    
    plt.plot(loss)
    plt.show()
    return point

if __name__=="__main__":
    print(main())