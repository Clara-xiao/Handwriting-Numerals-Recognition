# 手写数字识别 — 纯 NumPy 手搓 ANN

不用 TensorFlow / PyTorch，自己实现前向传播、反向传播和梯度下降，
在 MNIST (Kaggle Digit Recognizer) 上训练一个 **784 → 128 → 64 → 10** 的全连接网络，
训练完可以用来识别自己拍的手写数字照片。

## 网络结构

```
784 (28x28 像素)
  ↓  Z1 = X  @ W1 + b1    A1 = ReLU(Z1)
128
  ↓  Z2 = A1 @ W2 + b2    A2 = ReLU(Z2)
64
  ↓  Z3 = A2 @ W3 + b3    A3 = softmax(Z3)
10 (数字 0-9 的概率)
```

反向传播（输出层用 softmax + 交叉熵，梯度可以合并化简）：

```
dZ3 = (A3 - Y) / m                      输出层
dZl = (dZ_{l+1} @ W_{l+1}.T) * (Zl > 0)  隐藏层，ReLU 的导数
dWl = A_{l-1}.T @ dZl                    权重梯度
dbl = sum(dZl)                           偏置梯度
Wl -= lr * dWl                           梯度下降
```

W 和 b 只在开始随机初始化一次，之后**每个 mini-batch 原地更新一次**，
一路累积下去。20 个 epoch、batch 64，大约更新 10500 次。

## 文件结构

```
mnist-ann/
├── src/
│   ├── model.py       ANN 类：forward / backward / step / save / load + 梯度检验
│   └── preprocess.py  把手机照片转成 MNIST 格式 (28x28, 黑底白字, 居中)
├── train.py           训练并保存权重到 weights/mnist_ann.npz
├── predict.py         加载权重，识别自己拍的照片
├── submit.py          跑 Kaggle 测试集，生成 submission.csv
├── weights/           训练好的 W 和 b
└── samples/           放自己拍的照片
```

## 用法

```bash
pip install -r requirements.txt
```

### 1. 训练

在 Kaggle Notebook 里（数据已经挂载好）：

```bash
python train.py --plot
```

本地训练需要先从 Kaggle 下载 `train.csv`：

```bash
python train.py --data ./data/train.csv --plot
```

常用参数：

```bash
python train.py --epochs 30 --lr 0.05 --batch-size 32 --hidden1 256 --hidden2 128
```

训练时每个 epoch 评估一次，验证准确率创新高就自动保存权重。

### 2. 识别自己拍的照片

```bash
python predict.py samples/my_digit.jpg --show
```

`--show` 会用字符画把**模型真正看到的 28x28 图**打印出来。
预测错的时候先看这张图：如果它都不像数字（断笔、太细、有噪点），
那是预处理的问题，不是模型的问题。

拍照建议：白纸 + 粗黑笔（马克笔最好），一张图一个数字，光线均匀，数字占满画面。

### 3. 生成 Kaggle 提交文件

```bash
python submit.py
```

## 关于准确率

这个 ANN 在 MNIST 验证集上一般能到 **97% 左右**。

但在自己拍的照片上准确率会明显更低，这是正常的：全连接网络把图片拉成一长条，
丢掉了二维空间结构，对笔画位置、粗细、倾斜都很敏感。
想大幅提升，就要换成 CNN —— 这也正是学完 ANN 之后下一步要学的东西。

## 验证代码是否正确

```bash
python -c "import sys; sys.path.insert(0,'src'); from model import gradient_check; print(gradient_check())"
```

用数值梯度对比解析梯度，相对误差应该在 `1e-7` 量级。
注意这个检验内部用 float64 —— float32 精度不够，数值梯度自身的误差就能到 1e-2，
会看起来像是反向传播写错了。

## 可以继续做的实验

- 对比不同学习率 / 隐藏层大小 / batch size 对收敛的影响
- 加 L2 正则或 dropout，看过拟合有没有改善
- 把 ReLU 换成 sigmoid / tanh，对比收敛速度
- 实现 momentum 或 Adam，对比朴素梯度下降
- 做混淆矩阵，看看哪两个数字最容易混（通常是 4/9 和 3/5）
