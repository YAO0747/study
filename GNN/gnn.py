import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

# (node, label)集:[('n1', 0), ('n2', 0), ('n3', 0)....]
N = [("n{}".format(i), 0) for i in range(1, 7)] + \
    [("n{}".format(i), 1) for i in range(7, 13)] + \
    [("n{}".format(i), 2) for i in range(13, 19)]

# 边集
E = [("n1", "n2"), ("n1", "n3"), ("n1", "n5"),
     ("n2", "n4"),
     ("n3", "n6"), ("n3", "n9"),
     ("n4","n5"), ("n4", "n6"), ("n4", "n8"),
     ("n5","n14"),
     ("n7","n8"), ("n7", "n9"), ("n7", "n11"),
     ("n8","n10"), ("n8","n11"), ("n8", "n12"),
     ("n9","n10"), ("n9","n14"),
     ("n10","n12"),
     ("n11","n18"),
     ("n13","n15"), ("n13","n16"), ("n13","n18"),
     ("n14","n16"), ("n14","n18"),
     ("n15", "n16"), ("n15","n18"),
     ("n17", "n18")]


# 实现Xi函数，Xi输入的每一行：节点n的特征向量 + n的一个邻居u的特征向量
# ln是特征向量维度，s为状态向量维度
# Input : (N, 2*ln)
# Output : (N, S, S)
class Xi(nn.Module):
    def __init__(self, ln, s):
        super(Xi, self).__init__()
        self.ln = ln   # 节点特征向量的维度
        self.s = s     # 状态向量维度

        # 线性网络层
        self.linear = nn.Linear(in_features=2 * ln, out_features=s ** 2, bias=True)
        # 激活函数
        self.tanh = nn.Tanh()

    def forward(self, X):
        bs = X.size()[0]
        out = self.linear(X)
        out = self.tanh(out)
        return out.view(bs, self.s, self.s)


# 实现Rou函数,输入是节点n的特征向量，输出是节点n的状态向量
# Input : (N, ln)
# Output : (N, S)
class Rou(nn.Module):
    def __init__(self, ln, s):
        super(Rou, self).__init__()
        self.linear = nn.Linear(in_features=ln, out_features=s, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, X):
        return self.tanh(self.linear(X))


# 实现Hw函数：Hw的输入的每一行：节点n的特征向量 + n的一个邻居u的特征向量 + u的状态向量
# Input X: (N, 2 * ln): 每一行都是某个节点特征向量和该节点的某一邻居u的特征向量合并得到的向量
# Input H: (N, s)     : 对应中心节点的状态向量
# Input dg_list: (N, ): 对应中心节点的度的向量
# Output : (N, s)     : 神经网络Hw的输出，每一行代表：节点n的邻居节点u对节点n的状态的影响（最后要将n的所有邻居的影响进行求和）
class Hw(nn.Module):
    def __init__(self, ln, s, mu=0.9):
        super(Hw, self).__init__()
        self.ln = ln    # 节点特征向量的维度
        self.s = s      # 节点状态向量的维度
        self.mu = mu
        self.Xi = Xi(ln, s)
        self.Rou = Rou(ln, s)

    def forward(self, X, H, dg_list):
        if type(dg_list) == list:
            dg_list = torch.Tensor(dg_list)

        # dg_list.view(-1, 1, 1)是将长度为N的list转换为(N, 1, 1)的三维矩阵
        A = (self.Xi(X) * self.mu / self.s) / dg_list.view(-1, 1, 1)    # (N, S, S)
        b = self.Rou(torch.chunk(X, chunks=2, dim=1)[0])    # (N, S)
        # A:(N, S, S) * H: (N, S, 1) --> (N, S, 1) -->(squeeze)(N, S)
        out = torch.squeeze(torch.matmul(A, torch.unsqueeze(H, 2)), -1) + b
        return out    # (N, s)


# 求和：节点n的所有邻居节点的影响
class AggrSum(nn.Module):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.V = node_num

    def forward(self, H, X_node):
        # H : (N, s) -> (V, s)
        # X_node : (N, )
        mask = torch.stack([X_node] * self.V, 0)
        mask = mask.float() - torch.unsqueeze(torch.range(0, self.V-1).float(), 1)
        mask = (mask == 0).float()
        # (V, N) * (N, s) -> (V, s)
        return torch.mm(mask, H)


# 实现GNN模型
class OriLinearGNN(nn.Module):
    def __init__(self, node_num, feat_dim, stat_dim, T):
        super(OriLinearGNN, self).__init__()
        self.embed_dim = feat_dim       # ln = 2
        self.stat_dim = stat_dim        # s = 2
        self.T = T
        # 初始化节点的embedding，即节点特征向量 (V, ln)
        self.node_features = nn.Embedding(node_num, feat_dim)
        # 初始化节点的状态向量 (V, s)
        self.node_states = torch.zeros((node_num, stat_dim))
        self.linear = nn.Linear(feat_dim+stat_dim, 3)
        self.softmax = nn.Softmax()
        # 实现Hw
        self.Hw = Hw(feat_dim, stat_dim)
        # 实现H的分组求和
        self.Aggr = AggrSum(node_num)

    # X_Node类似于：[0, 0, 0, 1, 1, ..., 18, 18]，其中的值表示节点的索引，连续相同索引的个数为该节点的度
    # X_Neis类似于：[1, 2, 4, 1, 4, ..., 11, 13]，与X_Node一一对应，表示第一个向量节点的邻居节点
    # dg_list出度向量，例如A的出度数为3，B的出度数为2，则dg_list = [3,3,3,2,2,. . .]
    def forward(self, X_Node, X_Neis, dg_list):
        node_embeds = self.node_features(X_Node)  # (N, ln) (56, 2)
        neis_embeds = self.node_features(X_Neis)  # (N, ln) (56, 2)
        X = torch.cat((node_embeds, neis_embeds), 1)  # (N, 2 * ln)

        # 循环T次计算Hw(fw)
        for t in range(self.T):
            # (V, s) -> (N, s) :V是指节点数，而N是X_Node的长度
            # 例如node_states = [0.1, 0.2, 0.3], X_Node = (0, 0, 1, 1, 1, 2)则H = [0.1, 0.1, 0.2, 0.2, 0.2, 0.3]
            H = torch.index_select(self.node_states, 0, X_Node)
            # 训练Hw
            H = self.Hw(X, H, dg_list)
            self.node_states = self.Aggr(H, X_Node)     # (N, s) -> (V, s), 求n的所有邻居节点的影响之和，并更新状态向量

        # (gw)
        out = self.linear(torch.cat((self.node_features.weight, self.node_states), 1))
        out = self.softmax(out)
        return out


def CalAccuracy(output, label):
    out = np.argmax(output, axis=1)
    res = out - label
    return list(res).count(0) / len(res)


# 开始训练模型
def train(node_list, edge_list, label_list, T, ndict_path="./node_dict.json"):
    # 生成node-index字典
    if os.path.exists(ndict_path):
        with open(ndict_path, "r") as fp:
            node_dict = json.load(fp)
    else:
        node_dict = dict([(node, ind) for ind, node in enumerate(node_list)])
        node_dict = {"stoi" : node_dict, "itos" : node_list}
        with open(ndict_path, "w") as fp:
            json.dump(node_dict, fp)

    # 现在需要生成两个向量
    # 第一个向量node_inds类似于 [0, 0, 0, 1, 1, ..., 18, 18]
    # 其中的值表示节点的索引，连续相同索引的个数为该节点的度
    # 第二个向量node_neis类似于 [1, 2, 4, 1, 4, ..., 11, 13]
    # 与第一个向量一一对应，表示第一个向量节点的邻居节点

    # 首先统计得到节点的度
    Degree = dict()
    for n1, n2 in edge_list:
        # 边的第一个节点的邻接节点为第二个节点
        if n1 in Degree:
            Degree[n1].add(n2)
        else:
            Degree[n1] = {n2}
        # 边的第二个节点的邻接节点为第一个节点
        if n2 in Degree:
            Degree[n2].add(n1)
        else:
            Degree[n2] = {n1}

    # 然后生成两个向量
    node_inds = []
    node_neis = []
    for n in node_list:
        node_inds += [node_dict["stoi"][n]] * len(Degree[n])
        node_neis += list(map(lambda x: node_dict["stoi"][x], list(Degree[n])))
    # 生成度向量(出度数)，例如A的出度数为3，B的出度数为2，则dg_list = [3,3,3,2,2,. . .]
    dg_list = list(map(lambda x: len(Degree[node_dict["itos"][x]]), node_inds))

    # 准备训练集和测试集
    train_node_list = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    train_node_label = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    test_node_list = [3, 4, 5, 9, 10, 11, 15, 16, 17]
    test_node_label = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    # 开始训练
    model = OriLinearGNN(node_num=len(node_list), feat_dim=2, stat_dim=2, T=T)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(size_average=True)

    min_loss = float('inf')
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    node_inds_tensor = Variable(torch.Tensor(node_inds).long())
    node_neis_tensor = Variable(torch.Tensor(node_neis).long())
    train_label = Variable(torch.Tensor(train_node_label).long())
    for ep in range(200):
        # 运行模型得到结果,输入node_inds_tensor, node_neis_tensor, dg_list，forward前向传播得到res
        res = model(node_inds_tensor, node_neis_tensor, dg_list)   # (V, 3)

        # 提取出train_res和test_res
        train_res = torch.index_select(res, 0, torch.Tensor(train_node_list).long())
        test_res = torch.index_select(res, 0, torch.Tensor(test_node_list).long())

        # 计算训练集train_res的loss
        loss = criterion(input=train_res, target=train_label)
        loss_val = loss.item()

        # 计算准确率
        train_acc = CalAccuracy(train_res.cpu().detach().numpy(), np.array(train_node_label))
        test_acc = CalAccuracy(test_res.cpu().detach().numpy(), np.array(test_node_label))

        # 利用训练集的loss更新梯度
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # 保存loss和acc
        train_loss_list.append(loss_val)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        if loss_val < min_loss:
            min_loss = loss_val
        print("==> [Epoch {}] : loss {:.4f}, min_loss {:.4f}, train_acc {:.3f}, test_acc {:.3f}".format(ep, loss_val, min_loss, train_acc, test_acc))
    return train_loss_list, train_acc_list, test_acc_list


train_loss, train_acc, test_acc = train(node_list=list(map(lambda x:x[0], N)), edge_list=E, label_list=list(map(lambda x:x[1], N)), T=5)

