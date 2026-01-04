import torch
from torch.utils.cpp_extension import load_inline

# Build a tiny C++ extension that uses Vec256/Vectorized to square a float tensor.
source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/cpu/vec/vec.h>

torch::Tensor vec_square(torch::Tensor x) {
  auto out = torch::empty_like(x);
  // 为什么这里使用迭代器？解释下作用
  // TensorIterator 会对输入/输出张量做步长、广播、chunk 划分等元数据处理，
  // 统一生成遍历策略，后续 cpu_kernel_vec 按迭代器发出的块去做标量或矢量化计算。

  auto iter = at::TensorIterator::unary_op(out, x);

  // 这是什么写法？
  // cpu_kernel_vec 接收两个 lambda：第一个是标量版本（处理尾部不能整除向量宽度的元素），
  // 第二个是矢量化版本（使用 at::vec::Vectorized<T> 做 SIMD 运算）。框架会在支持的 ISA 下
  // 先用矢量 lambda 批量处理，再用标量 lambda 补齐剩余元素。

  at::native::cpu_kernel_vec(               // 矢量化的关键
      iter,
      [](float v) { return v * v; },
      [](at::vec::Vectorized<float> v) { return v * v; });     // 矢量化的关键
  return out;
}

// 如何绑定在 Python 的？解释 pybind 作用和用法
// PYBIND11_MODULE 声明一个名为 TORCH_EXTENSION_NAME 的 Python 模块；torch.utils.cpp_extension
// 会在编译时替换该宏为实际模块名。m.def 将 C++ 函数注册为 Python 可调用对象，
// 这样 load_inline 返回的扩展对象上就能通过 vec_square 访问此函数。

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vec_square", &vec_square, "Square using Vec256");
}
"""

vec_ext = load_inline(
    name="vec256_ext",
    cpp_sources=source,         # 源代码，就是上面的字符串，inline 就是在 python 里写字符串，编译成 cpp
    functions=None,
    verbose=False,          # 显示编译输出
)


def main() -> None:
    x = torch.arange(8, dtype=torch.float32)
    y = vec_ext.vec_square(x)
    print("x:", x)
    print("vec_square(x):", y)
    print("matches torch.square:", torch.allclose(y, torch.square(x)))


if __name__ == "__main__":
    main()
