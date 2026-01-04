import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

loss = Q.sum()
loss.backward()

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)
print(f"Q.grad: {Q.grad}")

# next_funcions是一个元组列表，元组内有两个参数，第一个参数是该节点的反向传播算子
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions)   # x对应的bwd算子

# 第二个参数是该节点来源于哪一个输入，这么说比较抽象，看例子吧……
a, b = torch.randn(2, requires_grad=True).unbind()
c = a+b
print(c.grad_fn.next_functions)