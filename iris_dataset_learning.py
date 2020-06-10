import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset=load_iris()
import TwolayerNet

net=TwolayerNet.TwoLayerNet(4,20,3,0.1)

#import iris data
#input data
x_data=iris_dataset["data"]
#teaching data,one hot lavel
target_list=iris_dataset["target"]
t_data=np.zeros((len(target_list),3))
for n in range(len(target_list)):
    t_data[n,target_list[n]]=1

#train data とtest dataに分割
X_train,X_test,t_train,t_test=train_test_split(x_data,t_data,random_state=0)

iter_num=4000
batch_size=20
lr=0.1 #学習率

#計算法選択　True:数値微分　False:誤差逆伝播
numerical=False

loss_list=[]
train_ac=[]
test_ac=[]
epoch=100

for i in range(iter_num):
    #バッチデータピックアップ
    batch_mask=np.random.choice(len(X_train),batch_size)
    x_batch=X_train[batch_mask]
    t_batch=t_train[batch_mask]

    #予測（順伝播）
    y=net.predict(x_batch)

    #数値微分により勾配（全微分）を算出。計算時間長大
    if numerical==True:
        accuracy=net.accuracy(y,t_batch)
        #グラフ表示まで待ちきれないので経過をコンソールに表示
        print(str(i)+":"+str(accuracy)+"%")
        #勾配の計算
        grads=net.num_grad(y,t_batch)

    #誤差逆伝播により勾配（全微分）を算出。計算時間大幅短縮
    else:
        loss=net.loss_cal(y,t_batch)
        accuracy=net.accuracy(y,t_batch)
        loss_list.append(loss)
        #勾配の計算
        dout=1
        grads=net.gradient(dout)

    #ネットワークパラメータの更新
    for key in ["W1","b1","W2","b2"]:
        net.params[key]-=lr*grads[key]
    
    #テストデータによる精度評価
    if i%epoch==0:
        #training dataのプロット
        train_ac.append(accuracy)
        #test dataのプロット
        y_test=net.predict(X_test)
        test_accuracy=net.accuracy(y_test,t_test)
        test_ac.append(test_accuracy)

#グラフ表示
plt.title("Accuracy of prediction of oreilly deep learning model :Iris dataset")
plt.plot(train_ac,label="train_data")
plt.plot(test_ac,linestyle='dashed',label="test_data")
plt.xlabel("epoch")
plt.ylabel("Accuracy(%)")
plt.legend()
plt.show()