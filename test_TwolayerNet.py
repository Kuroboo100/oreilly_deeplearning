import sys,os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

import TwolayerNet
net=TwolayerNet.TwoLayerNet(input_size=784,hidden_size=100,output_size=10,weight_init_std=0.1)
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

iter_num=10000
batch_size=100
lr=0.1 #学習率

#計算法選択　True:数値微分　False:誤差逆伝播
numerical=False

loss_list=[]
ac_list=[]
test_ac=[]
epoch=100

for i in range(iter_num):
    #バッチデータピックアップ
    batch_mask=np.random.choice(len(x_train),batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    #予測（順伝播）
    y=net.predict(x_batch)

    #数値微分により勾配（全微分）を算出。計算時間長大
    if numerical==True:
        accuracy=net.accuracy(y,t_batch)
        ac_list.append(accuracy)
        #グラフ表示まで待ちきれないので経過をコンソールに表示
        print(str(i)+":"+str(accuracy)+"%")
        #勾配の計算
        grads=net.num_grad(y,t_batch)

    #誤差逆伝播により勾配（全微分）を算出。計算時間大幅短縮
    else:
        loss=net.loss_cal(y,t_batch)
        accuracy=net.accuracy(y,t_batch)
        loss_list.append(loss)
        ac_list.append(accuracy)
        #勾配の計算
        dout=1
        grads=net.gradient(dout)

    #ネットワークパラメータの更新
    for key in ["W1","b1","W2","b2"]:
        net.params[key]-=lr*grads[key]
    
    #テストデータによる精度評価
    if i%epoch==0:
        y_test=net.predict(x_test)
        test_accuracy=net.accuracy(y_test,t_test)
        test_ac.append(test_accuracy)

#グラフ表示
#training data
plt.plot(ac_list)
plt.title("Accuracy(train_data)")
plt.xlabel("training time")
plt.ylabel("Accuracy(%)")
plt.show()

#test data
plt.plot(test_ac)
plt.title("Accuracy(test_data)")
plt.xlabel("epoch")
plt.ylabel("Accuracy(%)")
plt.show()


    

