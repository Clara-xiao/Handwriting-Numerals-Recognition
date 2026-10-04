"""
把手机拍的照片处理成 MNIST 的格式: 28x28, 黑底白字, 数字居中。

MNIST 原始的制作方式是: 把数字裁出来，最长边缩放到 20 像素，
再放进 28x28 的黑色画布中央。这里按同样的步骤做，
不然模型看到的输入和训练时差别太大，预测会不准。
"""
import numpy as np
from PIL import Image, ImageOps


def preprocess(path, invert=None, threshold=60):
    """
    参数:
        path      : 图片路径
        invert    : True = 反色(白纸黑字的照片要反色)
                    None = 自动判断
        threshold : 低于这个灰度值当作背景，去噪用
    返回:
        (28, 28) 的 float32 数组，值在 0~1
    """
    img = Image.open(path).convert("L")        # 转灰度
    img = ImageOps.autocontrast(img)           # 拉开对比度

    arr = np.array(img).astype(np.float32)
    if invert is None:
        # 边缘比中间亮，说明是白底黑字，需要反色
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        invert = border.mean() > 127
    if invert:
        arr = 255.0 - arr

    arr[arr < threshold] = 0                   # 去掉背景噪点
    if not arr.any():
        raise ValueError("图里没找到数字，试试换张对比度更高的照片")

    # 裁剪到数字的外接矩形
    ys, xs = np.where(arr > 0)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 最长边缩放到 20 像素，保持长宽比
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    # 贴到 28x28 黑色画布中央
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - new_w) // 2, (28 - new_h) // 2))

    return np.array(canvas, dtype=np.float32) / 255.0


def to_ascii(x28):
    """在终端里把 28x28 的图打印出来，方便确认预处理有没有出问题"""
    chars = " .:-=+*#%@"
    lines = []
    for row in x28:
        lines.append("".join(chars[min(int(v * len(chars)), len(chars) - 1)] for v in row))
    return "\n".join(lines)
