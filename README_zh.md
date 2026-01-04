# PyTorch 内部机制练习

一些用于学习 autograd、张量布局、分布式张量和 C++ 扩展的小脚本。每个文件都可以单独运行，方便把 Python API 和底层实现概念对照起来。

## 运行方式
- 在此目录下逐个运行：`python <脚本名>.py`
- 分布式示例 `dtensor.py` 用 torchrun 启动：`torchrun --nproc_per_node=<进程数> pytorch_internal/dtensor.py`
- `vector256_pytorch_cppextension.py` 会现场编译 C++ 扩展，需要本机有可用的编译器工具链。

## 脚本索引

### 自动求导与损失
- `cross_entropy_loss.py`: 最小化的交叉熵示例，包含 softmax 概率直觉和可求导输入。
- `torch_example.py`: 检查简单表达式的梯度，并查看 `grad_fn` / `next_functions` 以理解反向图。
- `torch_grad_sum.py`: 说明为何同一叶子张量的 `.grad` 会累加来自多条路径的梯度。
- `what_is_grad_and_gradfn.py`: 顺着 `grad_fn` 链路看幂运算的反向算子是如何记录的。

### 张量构造与内存布局
- `how_to_create_tensor_in_three_ways.py`: 三种生成多 rank 小输入的写法（循环、列表推导、张量向量化操作）。
- `tensor_view.py`: 通过切片/转置查看 stride、storage offset 和数据指针，理解视图如何共享底层存储。
- `tensor_sparse.py`: 构建 COO 稀疏张量，查看 `indices` / `values`，并与致密张量的 strided storage 对比。

### 分布式张量
- `dtensor.py`: 使用 `DeviceMesh`、`Shard`、`Replicate` 对逻辑张量做行分片与全量复制，对比各 rank 的本地张量并直接对 DTensor 计算。

### 扩展 PyTorch
- `vector256_pytorch_cppextension.py`: 内联 C++ 扩展，利用 `TensorIterator` 和 `cpu_kernel_vec` / `Vectorized` (Vec256) 做 SIMD 平方，并在 Python 中调用。
