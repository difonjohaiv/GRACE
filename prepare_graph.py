import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import csv
from tqdm import tqdm
import torch
from utils import get_labels


def get_graph():
    # 加载输入特征
    with open("content_embedding_m3e_base.pkl", 'rb') as f:
        features = pickle.load(f)

    # 加载标签
    labels = get_labels()
    y = torch.IntTensor(labels)

    # 读取CSV文件并解析内容
    with open('adj.csv', 'r') as file:
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
    edge_index = torch.tensor(edge_index_list,
                              dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    # 创建图对象
    graph = Data(x=features, edge_index=edge_index, edge_attr=edge_weight, y=y)

    # 有向到无向图
    undir_index, undir_weight = to_undirected(edge_index=graph.edge_index,
                                              edge_attr=graph.edge_attr)

    graph.edge_index, graph.edge_attr = undir_index, undir_weight

    print("是否为无向图:", graph.is_undirected())

    return graph
