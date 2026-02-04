import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm


# ----------------------------------------
# 1. 复用昨天的模拟器 (稍微精简一下作为函数)
# ----------------------------------------
def generate_sample(num_nodes=100):

    # A. 生成图
    G = nx.random_geometric_graph(num_nodes, 0.15)

    # B. 初始化属性
    for node in G.nodes():
        G.nodes[node]['load'] = np.random.uniform(1, 5)
        G.nodes[node]['capacity'] = G.nodes[node]['load'] * 1.6
        G.nodes[node]['status'] = 0  # 0 代表正常

    # 设置初始故障
    initial_failures = np.random.choice(G.nodes(), 3, replace=False)
    x = torch.zeros((num_nodes, 3), dtype=torch.float)
    for node in G.nodes():
        # 特征 0: 是否初始故障
        is_initial_fail = 1.0 if node in initial_failures else 0.0
        # 特征 1: 负载 (Load)
        load_val = G.nodes[node]['load']
        # 特征 2: 容量 (Capacity)
        cap_val = G.nodes[node]['capacity']

        x[node] = torch.tensor([is_initial_fail, load_val, cap_val])


    active_nodes = [n for n in G.nodes if G.nodes[n]['status'] == 0]
    while True:
        new_failures = []
        for node in active_nodes:
            neighbors = list(G.neighbors(node))
            failed_neighbors = [n for n in neighbors if G.nodes[n]['status'] == 1]
            if len(failed_neighbors) > 0:
                extra_load = len(failed_neighbors) * 0.8
                if G.nodes[node]['load'] + extra_load > G.nodes[node]['capacity']:
                    new_failures.append(node)

        if not new_failures:
            break
        for node in new_failures:
            G.nodes[node]['status'] = 1
        active_nodes = [n for n in G.nodes if G.nodes[n]['status'] == 0]

    # E. 收集标签 (y)
    y = torch.zeros(num_nodes, dtype=torch.long)
    for node in G.nodes():
        y[node] = G.nodes[node]['status']

    # F. 构建 PyG Data 对象
    # 获取边列表 (Edge Index)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 无向图

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


# GNN 模型 (GCN)
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(3, 16)

        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 卷积 + ReLU激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # 第二层卷积
        x = self.conv2(x, edge_index)

        # 输出概率分布
        return F.log_softmax(x, dim=1)

# 3. 主程序：生成数据 -> 训练
if __name__ == "__main__":
    # 批量生成场景
    print("正在生成 1000 个模拟场景作为训练数据 (Theory-Driven Data)...")
    dataset = []
    for _ in tqdm(range(1000)):
        data = generate_sample(num_nodes=100)
        dataset.append(data)

    # 划分训练集和测试集
    train_dataset = dataset[:400]
    test_dataset = dataset[400:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(f"🚀 模型已搭建，使用设备: {device}")

    # 训练循环
    print("开始训练 GNN...")
    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)

            # 计算 Loss: 预测值 vs 真实模拟结果
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")


    model.eval()
    correct = 0
    total_nodes = 0
    for data in test_dataset:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]  # 获取预测类别
        correct += pred.eq(data.y).sum().item()
        total_nodes += data.num_nodes

    acc = correct / total_nodes
    print(f"✅ GNN 预测准确率: {acc:.2%}")