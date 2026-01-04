"""
示例：为什么 Tensor.grad 要“累加”？

1) 同一个叶子张量在计算图里被用到多次（如 y = x^3 + x^2），反向时每条路径的梯度
   都会加到 .grad 上，结果就是所有贡献的和。
2) 如果把多个损失分开 backward（先 y1.backward，再 y2.backward），框架会把梯度累加到
   同一个 .grad，用来支持“多 loss 组合”或“梯度累积”。
"""

import torch


def multi_use_single_backward():
    # 同一图里的多条路径：y = x^3 + x^2
    x = torch.tensor(2.0, requires_grad=True)

    # 关键点：y = x**3 + x**2 里同一个叶子张量 x 在同一张计算图里走了两条路径（x→x^3 和 x→x^2）。
    # 反向传播时，来自两条路径对 x 的梯度贡献会在引擎内部做求和（chain rule 下的“多路径求和”），
    # 最终 x.grad = d(x^3)/dx + d(x^2)/dx = 3x^2 + 2x。
    # 这里的“累加”发生在一次 backward() 内部，是这个公式最对应的例子。
    y = x**3 + x**2
    y.backward()
    expected = 3 * x.detach() ** 2 + 2 * x.detach()  # 3x^2 + 2x
    print("单次 backward，自动把两条路径的梯度相加:")
    print("x.grad =", x.grad.item(), "预期 =", expected.item())  # 16


if __name__ == "__main__":
    multi_use_single_backward()
