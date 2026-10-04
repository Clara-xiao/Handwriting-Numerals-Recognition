"""
训练手写数字 ANN，并把学到的 W, b 保存成 weights/mnist_ann.npz。

用法:
    python train.py                                  # 默认参数
    python train.py --epochs 30 --lr 0.05            # 改超参数
    python train.py --data /kaggle/input/digit-recognizer/train.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import ANN, one_hot  # noqa: E402


def load_data(csv_path, val_ratio=0.2, seed=42):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values.astype(np.float32) / 255.0   # 归一化到 0~1
    y = df["label"].values.astype(np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(val_ratio * len(X))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def evaluate(model, X, y, Y):
    probs = model.forward(X)
    return model.loss(probs, Y), float((probs.argmax(1) == y).mean())


def train(args):
    X_train, y_train, X_val, y_val = load_data(args.data, args.val_ratio, args.seed)
    Y_train, Y_val = one_hot(y_train), one_hot(y_val)
    print(f"训练集 {X_train.shape[0]} 张, 验证集 {X_val.shape[0]} 张")

    model = ANN(sizes=(784, args.hidden1, args.hidden2, 10), seed=args.seed)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, n_batches = 0.0, 0
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(len(X_train))
        Xs, Ys = X_train[perm], Y_train[perm]

        # mini-batch 梯度下降: W 和 b 每个 batch 原地更新一次
        for start in range(0, len(Xs), args.batch_size):
            xb = Xs[start:start + args.batch_size]
            yb = Ys[start:start + args.batch_size]
            model.forward(xb)
            dWs, dbs = model.backward(yb)
            model.step(dWs, dbs, args.lr)
            n_batches += 1

        tr_loss, tr_acc = evaluate(model, X_train, y_train, Y_train)
        va_loss, va_acc = evaluate(model, X_val, y_val, Y_val)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | loss {tr_loss:.4f} | val_loss {va_loss:.4f} "
              f"| acc {tr_acc:.4f} | val_acc {va_acc:.4f} | {time.time()-t0:.1f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            model.save(args.out)

    print(f"\n一共更新了 {n_batches} 次 W 和 b")
    print(f"最好的验证准确率: {best_acc:.4f}")
    print(f"权重已保存到: {args.out}")

    if args.plot:
        plot_history(history)
    return model, history


def plot_history(history, path="training_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"曲线已保存到: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/kaggle/input/digit-recognizer/train.csv")
    p.add_argument("--out", default="weights/mnist_ann.npz")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--hidden1", type=int, default=128)
    p.add_argument("--hidden2", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="保存 loss/accuracy 曲线图")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
