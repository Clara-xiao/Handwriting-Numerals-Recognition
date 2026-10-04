"""
加载训练好的权重，识别你自己拍的手写数字照片。
不需要重新训练。

用法:
    python predict.py samples/digit.jpg
    python predict.py samples/*.jpg --show        # 打印模型看到的图
    python predict.py a.jpg --weights weights/mnist_ann.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import ANN            # noqa: E402
from preprocess import preprocess, to_ascii  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="一张或多张图片")
    p.add_argument("--weights", default="weights/mnist_ann.npz")
    p.add_argument("--show", action="store_true", help="用字符画打印预处理后的图")
    p.add_argument("--invert", choices=["auto", "yes", "no"], default="auto")
    args = p.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"找不到权重文件 {args.weights}，先运行 train.py")

    model = ANN.load(args.weights)
    invert = {"auto": None, "yes": True, "no": False}[args.invert]

    for path in args.images:
        try:
            x = preprocess(path, invert=invert)
        except Exception as e:
            print(f"{path}: 预处理失败 - {e}")
            continue

        probs = model.forward(x.reshape(1, 784))[0]
        pred = int(probs.argmax())
        top3 = np.argsort(probs)[::-1][:3]

        if args.show:
            print(to_ascii(x))
        print(f"{os.path.basename(path)} -> {pred}  (置信度 {probs[pred]:.1%})")
        print("   前三: " + ", ".join(f"{d}={probs[d]:.1%}" for d in top3))
        print()


if __name__ == "__main__":
    main()
