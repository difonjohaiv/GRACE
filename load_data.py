from utils import read_file
from utils import print_graph_detail
from utils import preprocess_adj
import networkx as nx
import torch as th
import pandas as pd
import numpy as np


def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {
                    "train", "training", "20news-bydate-train"
            }:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst


class PrepareData:

    def __init__(self, graph_path: str = "data/graph", dataset: str = "R8"):
        super(PrepareData, self).__init__()
        print("prepare data")
        self.graph_path = graph_path
        self.dataset = dataset

        # graph 利用边列表文件，构建带权重的图
        graph = nx.read_weighted_edgelist(
            f"{self.graph_path}/{self.dataset}.txt", nodetype=int)
        print_graph_detail(graph)
        # 将networkx中的图或边列表转换为scipy稀疏矩阵格式
        adj = nx.to_scipy_sparse_matrix(
            graph,
            nodelist=list(range(graph.number_of_nodes())),
            weight='weight',  # 表示为加权图
            dtype=float)
        # 矩阵对应元素相乘，multiply（条件）。不对称化邻接矩阵，利于GCN的训练
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.adj = preprocess_adj(adj, is_sparse=True)  # 邻接矩阵归一化

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # features
        self.nfeat_dim = graph.number_of_nodes()  #
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = th.from_numpy(np.vstack((row, col)).astype(np.int64))
        values = th.FloatTensor(value)
        shape = th.Size(shape)

        self.features = th.sparse.FloatTensor(indices, values, shape)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # target 数据真实标签

        target_fn = f"data/text_dataset/{self.dataset}.txt"  # 标签文件名
        target = np.array(pd.read_csv(target_fn, sep="\t",
                                      header=None)[2])  # pd中的一列
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # train val test split
        # get_train_test自定义方法
        self.train_lst, self.test_lst = get_train_test(target_fn)
