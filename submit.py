"""
用训练好的权重跑 Kaggle 测试集，生成 submission.csv。

用法:
    python submit.py
    python submit.py --data /kaggle/input/digit-recognizer/test.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import ANN  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/kaggle/input/digit-recognizer/test.csv")
    p.add_argument("--weights", default="weights/mnist_ann.npz")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=2048)
    args = p.parse_args()

    model = ANN.load(args.weights)
    X = pd.read_csv(args.data).values.astype(np.float32) / 255.0

    preds = []
    for start in range(0, len(X), args.batch_size):     # 分批，省内存
        preds.append(model.predict(X[start:start + args.batch_size]))
    preds = np.concatenate(preds)

    pd.DataFrame({"ImageId": np.arange(1, len(preds) + 1), "Label": preds}) \
        .to_csv(args.out, index=False)
    print(f"{len(preds)} 条预测已写入 {args.out}")


if __name__ == "__main__":
    main()
