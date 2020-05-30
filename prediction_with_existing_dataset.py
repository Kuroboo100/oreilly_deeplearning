import numpy as np
import sys,os
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
#各数字の予測正答率をグラフ化

def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
    return x_test,t_test

def init_network():
    with open("/home/yuki/anaconda3/python_programs/oreilly_deeplearning/deep-learning-from-scratch-master/ch03/sample_weight.pkl","rb") as f:
        network=pickle.load(f)
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    x=list(map(lambda e:e-np.max(x),x))
    exp_x=list(map(lambda e:np.exp(e),x))
    sum_expx=np.sum(exp_x)
    return exp_x/sum_expx

def predict(x,network):
    w1,w2,w3=network["W1"],network["W2"],network["W3"]
    b1,b2,b3=network["b1"],network["b2"],network["b3"]

    z1=sigmoid(np.dot(x,w1)+b1)
    z2=sigmoid(np.dot(z1,w2)+b2)
    z3=sigmoid(np.dot(z2,w3)+b3)
    return np.argmax(z3)

def cal_each_accuracy():
    x_test,t_test=get_data()
    network=init_network()
    N=len(x_test)
    accuracy_cnt=[0 for n in range(N)]
    appear_cnt=[0 for n in range(N)]
    for i in range(N):
        if  predict(x_test[i],network)==t_test[i]:
            accuracy_cnt[t_test[i]]+=1
            appear_cnt[t_test[i]]+=1
        else:
            appear_cnt[t_test[i]]+=1
    return [accuracy_cnt[t_test[i]]/appear_cnt[t_test[i]]*100 for i in range(10)]

def graph_plot(x,y):
    plt.bar(x,y)
    plt.show()

def main():
    x=np.arange(0,10,1)
    y=cal_each_accuracy()
    graph_plot(x,y)

if __name__=="__main__":
    main()



