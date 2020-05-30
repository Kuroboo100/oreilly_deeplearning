import numpy as np
import matplotlib.pyplot as plt

def dual(x1,x2):
    """
    入力：x
    出力：2変数多項式、係数、次数のリスト
    """
    #係数
    a=[1,1]
    #次数
    k=[2,2]
    return a[0]*x1**k[0]+a[1]*x2**k[1],a,k

def gradient(f,x):
    """
    入力：多項式関数、x(リスト)
    出力：各x(i,j)毎の全微分ベクトル(リスト)、多項式関数の係数、次数
    """
    d=10e-2
    grad_x1=np.zeros((len(x[0]),len(x[1])))
    grad_x2=np.zeros((len(x[0]),len(x[1])))
    a,k=f(0,0)[1:]

    for i in range(len(x[0])):
        for j in range(len(x[1])):
            #x[0]成分の偏微分
            dx1_p=f(x[0][i]+d,x[1][j])[0]
            dx1_m=f(x[0][i]-d,x[1][j])[0]
            ddx1=(dx1_p-dx1_m)/2*d
            #x[1]成分の偏微分
            dx2_p=f(x[0][i],x[1][j]+d)[0]
            dx2_m=f(x[0][i],x[1][j]-d)[0]
            ddx2=(dx2_p-dx2_m)/2*d
            
            grad_x1[i,j]=ddx1
            grad_x2[i,j]=ddx2
    return grad_x1,grad_x2,a,k

def graph(X,Y,U,V,a,k):
    """
    ベクトル場のプロット
    """
    plt.title("{}*x1^{}+{}*x2^{}".format(a[0],k[0],a[1],k[1]))
    plt.quiver(X,Y,U,V,color='red',angles='xy',scale_units='xy', scale=5.0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    return

def main():
    #ベクトル表示倍率
    mag=50
    #x1,x2の定義
    x=[np.arange(-10,10,1),np.arange(-10,10,1)]
    #数値微分
    U,V,a,k=gradient(dual,x)
    x_=[]
    y_=[]
    u=[]
    v=[]

    for i in range(len(x[0])):
        for j in range(len(x[1])):
            x_.append(x[0][i])
            y_.append(x[1][j])
            u.append(U[i,j]*mag)
            v.append(V[i,j]*mag)
    graph(x_,y_,u,v,a,k)
    return

if __name__=="__main__":
    main()

