import torch
import torch.nn as nn

# 输入和目标
input = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True)
target = torch.tensor([0, 1])

# 使用 CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
"""
1. 输入softmax归一化，得到[0.659, 0.242, 0.098] [0.159, 0.711, 0.130]
2. target label是【0,1】意味着第一个样本是第0类，第二个样本是第1类
3. 第一个样本的loss(第0类) l1 = log(0.659) = 0.417; 第二个样本的loss l2 = -log(0.711) = 0.343
4. 最终loss = (l1+l2)/2=0.38
"""
loss = criterion(input, target)

# 输出损失
print(loss.item())

