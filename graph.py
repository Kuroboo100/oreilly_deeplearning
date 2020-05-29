import matplotlib.pyplot as plt
import numpy as np

def step(x):
    y=x>0
    return y.astype(int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def graphplot(x,y1,y2):
    fig,ax=plt.subplots()
    ax.set_title("function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(x,y1,color="red")
    ax.plot(x,y2,color="blue",linestyle="dashed")
    plt.show()
    return

def main():
    x=np.arange(-10,10,0.01)
    y1=step(x)
    y2=sigmoid(x)
    graphplot(x,y1,y2)
    return

if __name__=="__main__":
    main()