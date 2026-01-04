# PyTorch Internals Playground

[中文版本](README_zh.md)

A collection of tiny, focused PyTorch scripts I wrote while studying autograd, tensor layouts, distributed tensors, and C++ extensions. Inspired by [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/), these snippets stay small and runnable so it is easy to map the Python surface API back to the underlying mechanics.

## How to run
- Clone/open this folder and run individual snippets with `python <script>.py`.
- For distributed `dtensor.py`, launch with `torchrun --nproc_per_node=<ranks> pytorch_internal/dtensor.py`.
- `vector256_pytorch_cppextension.py` builds a C++ extension on the fly; you need a compiler toolchain that can compile against your local PyTorch headers.

## Script guide

### Autograd & loss essentials
- `cross_entropy_loss.py`: Minimal CrossEntropyLoss example with manual softmax/prob intuition and a gradient-enabled input.
- `torch_example.py`: Checks gradients of simple expressions and inspects `grad_fn`/`next_functions` to see the backward graph.
- `torch_grad_sum.py`: Shows why `.grad` accumulates contributions when the same leaf tensor flows through multiple graph paths.
- `what_is_grad_and_gradfn.py`: Traverses `grad_fn` chains for elementwise power operations to expose how autograd records backward ops.

### Tensor construction & memory layout
- `how_to_create_tensor_in_three_ways.py`: Three ways to generate per-rank toy inputs (loops, list comprehension, vectorized tensor ops).
- `tensor_view.py`: Investigates views, strides, storage offsets, and raw data pointers to show how slicing/transposing reuses storage.
- `tensor_sparse.py`: Builds a COO sparse tensor, inspects `indices`/`values`, and contrasts sparse storage with dense strided storage.

### Distributed tensors
- `dtensor.py`: Uses `DeviceMesh`, `Shard`, and `Replicate` to shard a logical tensor across ranks, compare local shards vs replicated tensors, and run ops directly on `DTensor`.

### Extending PyTorch
- `vector256_pytorch_cppextension.py`: Inline C++ extension that uses `TensorIterator` + `cpu_kernel_vec`/`Vectorized` (Vec256) to SIMD-square a float tensor, then calls it from Python.

Filenames are now standardized to snake_case for easier linking from the blog.
