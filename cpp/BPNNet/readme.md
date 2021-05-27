# BP神经网络
[TOC]

## 使用方法

所有内容封装在BPNNet类中

调用流程:
1. 以网络层数、各层节点数为参数创建BPNNet对象
2. 网络初始(必要，否则无法传播)
3. 网络训练
4. 向前传播
```c++
int layers[] = {node1, node2, node3, node4};
BPNNet net(3, layers);
net.Initial();
int times = 500000;
net.Train(trainInput, trainTarget, ptsNum, times);
net.ForePropagate(x);
double *result = new double[node4];
result = net.output;
```

## 提交记录

### 第一次提交(21/05/23)

实现了BP神经网络，但存在问题：

网络的初始化会严重影响到网络训练结果，其中网络的```weight```和```bias```都是使用c++的随机数引擎```default_random_engine```生成的，经过测试

1. 网络的传播应该没有问题，因为在若干次训练中结果正确的几率不小
2. 当随机数引擎不设置种子时(自动按照时间设置？)，每次训练的结果存在差异，多次测试结果中会出现正确或错误的情况，初步判断应该是网络初始化的影响
3. 当固定随机数种子时，训练结果完全一致(存疑)

### 第二次提交(21/05/24)

小修改了一下，BPNNet.h中定义的全局随机数引擎
```c++
static std::default_random_engine randomEngine;
```
在网络的整个流程中使用，包括了初始化和训练选样本
此外还改了点细枝末节，但是为啥突然就好了呢？？？？？

