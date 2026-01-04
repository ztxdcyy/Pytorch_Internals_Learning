import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个宏
# define

# # 创建模型并进行前向传播
# x = torch.tensor([1.], requires_grad=True)
# m = torch.tensor([1, 0], dtype=torch.float32, requires_grad= True)
# y = 3 * x + 2*m     
# z = y * 2 + m
# loss = (z - 1).pow(2).sum()  # 计算损失
# loss.backward()

# print(f"Loss grad fn is: {loss.grad_fn}")                                   # loss 是由sum得到的，所以这里是sumbwd0
# print(f"Loss's next_fn is: {loss.grad_fn.next_functions[0][0]}")            # loss的上一步是(z-1)^2，把z-1看成u，也就是u^2，所以loss的next_fn是powbwd0    
# print(f"Loss's next_fn is: {loss.grad_fn.next_functions[0][0].next_functions[0][0]}")       # u = z-1
# print(f"Loss's next_fn is: {loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0]}")  # z = n+m
# print(f"Loss's next_fn is: {loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0]}")     # n = 2*y
# print(f"Loss's next_fn is: {loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[1][0]}")     # m AccumlatedGrad
# print(x.grad if x.grad is not None else "None!!!")

x = torch.tensor([1.,2.], requires_grad = True)
y = torch.tensor([3., 4.], requires_grad= True)
z = 3 * x**2 + x**3
# z = x + y
loss = z.sum()
loss.backward()

print(x.grad)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions)   # x对应的bwd算子

# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[1][0])
# print(loss.grad_fn.next_functions[1][0])


a, b = torch.randn(2, requires_grad=True).unbind()
c = a+b
print(c.grad_fn.next_functions)