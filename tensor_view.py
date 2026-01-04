import torch

A= torch.arange(4, dtype=torch.int32).view(2, 2)
A = torch.tensor([1024, 32, 16, 10], dtype=torch.int32).view(2,2)
B = A[1, ]          

print("A:\n", A)
print("A.size():", A.size())
print("A.stride():", A.stride())
print()

print("B:\n", B)
print("B.size():", B.size())
print("B.stride():", B.stride())
print("B's offset: ", B.storage_offset())
print()

C = A.t()
print(C)
print(C.stride(), "\n")             # 新增一个例子用于理解stride


print(A.untyped_storage(), "\n")    

# 对比底层 untyped storage 的地址
print("A & B same storage:",
      A.untyped_storage().data_ptr() == B.untyped_storage().data_ptr())
print("A & C same storage:",
      A.untyped_storage().data_ptr() == C.untyped_storage().data_ptr())

print(f"A.data_ptr(): {A.untyped_storage().data_ptr()}")
print(f"B.data_ptr(): {B.untyped_storage().data_ptr()}")
print(f"A.element_size(): {A.element_size()}")
print(f"A.storage_offset(): {A.storage_offset()}")
print(f"B.storage_offset(): {B.storage_offset()}")

# 计算实际地址
A_actual = A.untyped_storage().data_ptr() + A.storage_offset() * A.element_size()
B_actual = B.untyped_storage().data_ptr() + B.storage_offset() * B.element_size()

print(f"A第一个元素实际地址: {A_actual}")
print(f"B第一个元素实际地址: {B_actual}")
print(f"地址差: {B_actual - A_actual} 字节 (应该等于 2 * 4 = 8 字节)")
