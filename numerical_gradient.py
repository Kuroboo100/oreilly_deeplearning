import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return x[0]**2+x[1]**2

def gradient(f,x):
    """
    ある一点の座標xにおける各成分を数値偏微分し、全微分ベクトルを求める関数。
    入力：座標
    出力：入力座標に対する全微分ベクトル
    """
    h=10e-4
    grad=np.zeros_like(x)
    for idx in range(len(x)):
        tmp=x[idx]
        x[idx]=tmp+h
        fx_p=f(x)

        x[idx]=tmp-h
        fx_m=f(x)
        grad[idx]=((fx_p-fx_m)/(2*h))
        x[idx]=tmp
    return grad

def graph(x,y,u,v):
    plt.quiver(x,y,u,v)
    plt.show()
    return

def main():
    #座標の定義
    x=[]
    x_=np.arange(-10,10,1)
    y_=np.arange(-10,10,1)
    x=np.meshgrid(x_,y_)
    x1=x[0].ravel()
    x2=x[1].ravel()
    #各座標における勾配ベクトルを算出し、u成分とv成分に分けて格納
    u=[]
    v=[]
    for n in range(len(x1)):
        tmp=-gradient(function,[x1[n],x2[n]])
        u.append(tmp[0])
        v.append(tmp[1])
    graph(x1,x2,u,v)
    return

if __name__=="__main__":
    main()



        