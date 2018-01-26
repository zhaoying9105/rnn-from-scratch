# Softmax

## predict
这个softmax比较水，只是求指数后归一化
输出的结果是个向量，每个位置的值代表一个词汇的可能性
## loss
输入的y 是个标量，代表索引。比如`y = 6`，则词汇表中的第6个词汇，同时probs[6] 是 0.3，那么 `-np.log(probs[y]) ` = `-np.log(0.3)`。当`probs[y]`越小时，损失越大

## diff
求微分

同样假设`y = 6`，同时`probs[6]` 是 `0.3`，那么 `probs[y] -= 1.0` = `-0.7`
当`probs[y]`越小时，损失越大

### 数学推导

这段代码写的应该是有问题的

$$\hat{y} = e^x$$
$$loss = - \log (\hat{y} - y)$$
$y$可以是 0 或者 1

1. 当 y = 0

$$loss = - \log (\hat{y})$$
$$\frac{\partial loss}{\partial x}  = \frac{\partial loss}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial x}$$

$$ = - \frac{1}{\hat{y}} \cdot \hat{y}$$

$$= -1 $$


1. 当 y = 1

$$loss = - \log (\hat{y} -1)$$
$$\frac{\partial loss}{\partial x}  = \frac{\partial loss}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial x}$$

$$ = - \frac{1}{\hat{y} -1} \cdot \hat{y}$$

$$= \frac{e^x}{e^x -1} $$

