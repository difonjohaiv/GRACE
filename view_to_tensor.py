import csv
import torch
from torch_sparse import coalesce
from tqdm import tqdm

# 读取CSV文件并解析内容
with open('thucnews_title.csv', 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    content = list(reader)

# 创建空列表
edge_index_list = []
edge_weight_list = []

# 遍历内容列表并添加到相应的列表中
for row in tqdm(content):
    index1 = int(row[0])
    index2 = int(row[1])
    weight = float(row[2])
    edge_index_list.append([index1, index2])
    edge_weight_list.append(weight)

# 创建输入张量
edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

# 获取稀疏张量的形状
m = edge_index.max().item() + 1
n = edge_index.max().item() + 1

# 将边的索引和权重转换为稀疏张量，非必要
edge_index, edge_weight = coalesce(edge_index, edge_weight, m=m, n=n)

# 打印结果
print("Edge Index:\n", edge_index)
print("Edge Weight:\n", edge_weight)
