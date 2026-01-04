import torch

# 方法1：使用循环创建虚拟rank的输入
def create_input_with_loops(num_ranks=4):
    inputs = []
    for rank in range(num_ranks):
        # 为每个rank创建输入数据
        input_data = torch.arange(4) + rank * 4
        inputs.append(input_data)
    return inputs

# 方法2：使用列表推导式（更简洁）
def create_input_with_list_comprehension(num_ranks=4):
    return [torch.arange(4) + rank * 4 for rank in range(num_ranks)]

# 方法3：使用torch张量操作（更高效）
def create_input_with_tensor_ops(num_ranks=4):
    # 创建形状为[num_ranks, 4]的张量
    ranks = torch.arange(num_ranks).view(-1, 1)
    # 为每个rank生成4个连续的数
    return torch.arange(4) + ranks * 4

# 示例使用
print("方法1结果:")
result1 = create_input_with_loops()
for i, data in enumerate(result1):
    print(f"Rank {i}: {data}")

print("\n方法2结果:")
result2 = create_input_with_list_comprehension()
for i, data in enumerate(result2):
    print(f"Rank {i}: {data}")

print("\n方法3结果:")
result3 = create_input_with_tensor_ops()
print(result3)