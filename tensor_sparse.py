"""
PyTorch 稀疏张量示例，以及稀疏张量的底层存储（indices + values）与常规
stride 张量（Storage + stride 描述同一块连续内存）的区别。
"""

import torch


def sparse_example():
    # COO 形式的稀疏张量：只存非零的坐标和对应的值。
    indices = torch.tensor([[0, 0, 1], [1, 2, 0]])  # 形状 [2, nnz]，行坐标和列坐标
    values = torch.tensor([10.0, 20.0, 30.0])  # 非零值
    size = (2, 3)

    sp = torch.sparse_coo_tensor(indices, values, size).coalesce()  # coalesce 合并重复坐标
    print("稀疏张量 sp:\n", sp)
    print("稀疏底层 indices:\n", sp.indices())  # 只保存非零坐标
    print("稀疏底层 values:\n", sp.values())  # 只保存非零值

    dense = sp.to_dense()
    print("转成致密张量:\n", dense)
    return sp, dense


def compare_to_strided(sp, dense):
    # 常规张量使用一块连续 Storage，加 stride 描述步幅来解释同一块内存。
    print("致密 storage 元素个数:", dense.storage().size())
    print("致密 stride:", dense.stride())

    # 转置后 stride 改变，但 storage 复用同一块内存。
    transposed = dense.t()
    print("转置后 stride:", transposed.stride())
    print("storage 是否相同:", dense.storage().data_ptr() == transposed.storage().data_ptr())

    # 展平后的元素顺序对应 storage 的线性布局。
    print("致密展平元素顺序:", dense.view(-1).tolist())

    # 稀疏张量不存储零值，仅记录 nnz 个 values，对比总元素数。
    print("稀疏存储的非零元素个数:", sp.values().numel(), " / 总大小:", sp.numel())


if __name__ == "__main__":
    sp, dense = sparse_example()
    compare_to_strided(sp, dense)
