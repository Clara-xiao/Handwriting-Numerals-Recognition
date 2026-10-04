"""
纯 NumPy 实现的全连接神经网络 (ANN)。

每一层做的事:
    Z = A_prev @ W + b
    A = ReLU(Z)          (隐藏层)
    A = softmax(Z)       (输出层)

反向传播:
    输出层 (softmax + cross-entropy):  dZ = (A - Y) / m
    隐藏层:                            dZ = (dZ_next @ W_next.T) * (Z > 0)
    梯度:                              dW = A_prev.T @ dZ ,  db = sum(dZ)
"""
import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)      # 减最大值，防止 exp 溢出
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(labels, num_classes=10):
    out = np.zeros((len(labels), num_classes), dtype=np.float32)
    out[np.arange(len(labels)), labels] = 1.0
    return out


class ANN:
    """默认结构 784 -> 128 -> 64 -> 10"""

    def __init__(self, sizes=(784, 128, 64, 10), seed=42):
        self.sizes = tuple(sizes)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            # He initialization，配合 ReLU 使用
            w = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
            self.W.append(w.astype(np.float32))
            self.b.append(np.zeros((1, n_out), dtype=np.float32))

    @property
    def n_layers(self):
        return len(self.W)

    # ---------------- 前向传播 ----------------
    def forward(self, X):
        """返回 (batch, 10) 的概率，并缓存中间结果供 backward 使用"""
        self.acts = [X]      # A0 = X, A1, ..., A_L
        self.zs = []         # Z1, ..., Z_L
        A = X
        for i in range(self.n_layers):
            Z = A @ self.W[i] + self.b[i]
            self.zs.append(Z)
            A = softmax(Z) if i == self.n_layers - 1 else relu(Z)
            self.acts.append(A)
        return A

    # ---------------- 损失 ----------------
    @staticmethod
    def loss(probs, Y):
        """交叉熵损失。probs 是 softmax 输出，Y 是 one-hot 标签"""
        return float(-np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1)))

    # ---------------- 反向传播 ----------------
    def backward(self, Y):
        """必须先调用 forward。返回每一层的 dW, db"""
        m = Y.shape[0]
        dWs = [None] * self.n_layers
        dbs = [None] * self.n_layers
        dZ = (self.acts[-1] - Y) / m                      # softmax + CE 合并后的梯度
        for i in reversed(range(self.n_layers)):
            dWs[i] = self.acts[i].T @ dZ
            dbs[i] = dZ.sum(axis=0, keepdims=True)
            if i > 0:
                dZ = (dZ @ self.W[i].T) * (self.zs[i - 1] > 0)   # ReLU 的导数
        return dWs, dbs

    # ---------------- 梯度下降 ----------------
    def step(self, dWs, dbs, lr):
        for i in range(self.n_layers):
            self.W[i] -= lr * dWs[i]
            self.b[i] -= lr * dbs[i]

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    # ---------------- 保存 / 加载 ----------------
    def save(self, path):
        arrays = {"sizes": np.array(self.sizes)}
        for i in range(self.n_layers):
            arrays[f"W{i}"] = self.W[i]
            arrays[f"b{i}"] = self.b[i]
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(sizes=tuple(int(s) for s in d["sizes"]))
        for i in range(model.n_layers):
            model.W[i] = d[f"W{i}"]
            model.b[i] = d[f"b{i}"]
        return model


def gradient_check(seed=0):
    """
    用数值梯度验证 backward 写得对不对。相对误差应该在 1e-7 量级。

    注意: 这里必须用 float64。float32 精度不够，
    数值梯度本身的误差就能到 1e-2，看起来像是反向传播写错了。
    """
    rng = np.random.default_rng(seed)
    model = ANN(sizes=(5, 4, 3), seed=seed)
    model.W = [w.astype(np.float64) for w in model.W]
    model.b = [b.astype(np.float64) for b in model.b]
    X = rng.standard_normal((6, 5))
    Y = one_hot(rng.integers(0, 3, 6), 3).astype(np.float64)

    model.forward(X)
    dWs, _ = model.backward(Y)

    eps = 1e-6
    worst = 0.0
    for layer in range(model.n_layers):
        for _ in range(8):
            i = rng.integers(0, model.W[layer].shape[0])
            j = rng.integers(0, model.W[layer].shape[1])
            original = model.W[layer][i, j].copy()

            model.W[layer][i, j] = original + eps
            loss_plus = model.loss(model.forward(X), Y)
            model.W[layer][i, j] = original - eps
            loss_minus = model.loss(model.forward(X), Y)
            model.W[layer][i, j] = original

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = dWs[layer][i, j]
            denom = max(abs(numeric) + abs(analytic), 1e-8)
            worst = max(worst, abs(numeric - analytic) / denom)
    return worst
