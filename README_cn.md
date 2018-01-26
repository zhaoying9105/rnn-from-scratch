# 从零开始实现 循环神经网络

这里假定读者已经熟悉基础的神经网络，如果没有的话，可以移步[从零开始实现神经网络( Implementing A Neural Network From Scratch)]( Implementing A Neural Network From Scratch)，这个项目会向你讲解非循环网络的概率和实现

## 介绍

这篇文章是受 [WildML](http://www.wildml.com/) 的 [recurrent-neural-networks-tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) 项目启发，你可以自己深入研究这个教程， 这里我们不多做介绍

在我们的教程中，我们关注如何在**计算图**和**自动微分**的基础上通过 [沿着时间的反向传播（Backpropagation Through Time (BPTT)）](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)来训练RNN。你会发现这比你纯手动计算梯度要更简单可靠

本文用RNN 语言模型作为例子，更有有趣的RNN 应用[这里](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).


## 如何训练RNN
RNN结构如下
![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn.jpg)

我们发现参数 `(W, U, V)` 在不同的时间步中是共享的，每个时间步的输出是`softmax`，所有你可以使用**交叉熵**作为损失函数，然后使用一些优化方法（比如梯度下降）来优化参数 `(W, U, V)`。

我们先简要回顾下RNN 的公式

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn_equation.png)

我们也定义一个损失函数，这里使用交叉熵

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn_loss.png)

这里，$h_t$ 是t 时间步的正确的词汇，$\hat{\h_t}$是我们预测的词汇，我们使用整个序列（一个句子）作为一个训练样本，所有总的损失值是每个时间步（词汇）的损失值之和

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn-bptt1.png)

记住，我们的目标是计算参数 `(W, U, V)` 的梯度，然后使用优化方法（这篇文章里我们使用**随机梯度下降**）来学习到好的参数。就像我们对时间步的损失值进行加和，我们计算梯度时也把把一个训练样本中的不同时间步的梯度加起来 ![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/gradient.png)。就是说我们计算 $\frac{\partial E_t}{\partial W}$， $\frac{\partial E_t}{\partial U}$， $\frac{\partial E_t}{\partial V}$，然后把不同时间步的梯度加起来。

计算 $\frac{\partial E_t}{\partial V}$ 很简单，因为它只依赖当前时间步的值。

但$\frac{\partial E_t}{\partial W}$，$\frac{\partial E_t}{\partial U}$就不一样了。比如$s_3 = \tanh(U \times x_3 + W \times s_2)$，$s_3$依赖$s_2$，$s_2$又依赖$U$,$W$,$s_1$等等，如果我们对$W$求微分，我们却不能把$s_2$看做常数，我们又要应用链式法则，情况如下图所示：
![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn-bptt-with-gradients.png)


现在我们做个例子，使用计算图来表示$E_1$和计算$\frac{\partial E_t}{\partial W}$，$\frac{\partial E_t}{\partial U}$

![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn-compuattion-graph.png)

注意：这里和我们熟悉的前馈神经网络输出完全一致的，关键的区别在与我们把每步$W$的梯度加起来了。在传统的神经网络里，我们不会跨层共享参数，所以我们没有任何求和操作。

为了简化计算图，我们把几个操作整合起来用一个大的操作来表示它们。请看下图，注意下面的操作节点也都要是吗前向和反向传播
![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/rnn-compuattion-graph_2.png)

所有操作节点和softmax输出的实现都现在下面

```python
mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class RNNLayer:
    def forward(self, x, prev_s, U, W, V):
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)
        
    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)
```

```python
class MultiplyGate:
    def forward(self,W, x):
        return np.dot(W, x)
    def backward(self, W, x, dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx

class AddGate:
    def forward(self, x1, x2):
        return x1 + x2
    def backward(self, x1, x2, dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2
```

```python
class Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - output) * output * top_diff

class Tanh:
    def forward(self, x):
        return np.tanh(x)
    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff
```

```python
class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)
    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])
    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs
```

以上的实现和[Implementing A Neural Network From Scratch](https://github.com/pangolulu/neural-network-from-scratch)完全一样，除了这里的输入x或者s 是 一维向量，但是在那个项目中是按批次排列的矩阵

现在我们可以通过随机梯度下降来计算参数了

## 实现


## 初始化
先用一个小技巧初始`W U V`三个参数，我们不能直接初始化为0，这会导致我们所有层的计算结果都是对称的，我们必须随机初始化。最佳的初始化策略和激活函数有关，有人推荐以![](https://github.com/pangolulu/rnn-from-scratch/raw/master/figures/init.png) 为间隔随机初始化，其中n 是输入的长度
```python
class Model:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
```

