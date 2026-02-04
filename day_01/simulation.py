import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from datetime import datetime
from tqdm import tqdm


class CascadingSimulator:
    def __init__(self, num_nodes=100, radius=0.15):
        """
        初始化一个具有空间属性的图 (模拟基础设施网络)
        使用 Random Geometric Graph，因为现实中的电网/路网都是受地理距离限制的
        """
        self.num_nodes = num_nodes
        # 生成带空间坐标的图结构
        self.G = nx.random_geometric_graph(num_nodes, radius)

        # 初始化节点物理属性
        # Load: 当前负载;
        # Capacity: 最大承受阈值
        for node in self.G.nodes():
            self.G.nodes[node]['load'] = np.random.uniform(1, 5)  # 初始负载
            self.G.nodes[node]['capacity'] = self.G.nodes[node]['load'] * 1.5
            self.G.nodes[node]['status'] = 'active'  # 状态

    def trigger_failure(self, num_initial_failures=2):
        # 重置图的状态
        for node in self.G.nodes():
            self.G.nodes[node]['status'] = 'active'

        # 随机选择初始故障点
        initial_targets = random.sample(list(self.G.nodes()), num_initial_failures)
        for node in initial_targets:
            self.G.nodes[node]['status'] = 'failed'

        return initial_targets

    def run_cascade(self):

        # 级联失效传播
        graph = self.G.copy()
        active_nodes = [n for n in graph.nodes if graph.nodes[n]['status'] == 'active']

        step = 0
        cascade_history = []

        while True:
            new_failures = []

            # 找到所有失效的节点
            failed_nodes = [n for n in graph.nodes if graph.nodes[n]['status'] == 'failed']

            # 检测是否有新的过载节点
            for node in active_nodes:
                if graph.nodes[node]['status'] == 'failed':
                    continue

                # 计算这个节点是否过载
                neighbors = list(graph.neighbors(node))
                failed_neighbors = [n for n in neighbors if graph.nodes[n]['status'] == 'failed']

                # 冲击逻辑：每个挂掉的邻居会给你带来额外的 1.0 压力
                extra_load = len(failed_neighbors) * 1.5
                current_load = graph.nodes[node]['load'] + extra_load

                if current_load > graph.nodes[node]['capacity']:
                    new_failures.append(node)

            # 如果这一轮没有新节点挂掉，级联结束
            if not new_failures:
                break

            # 更新状态
            for node in new_failures:
                graph.nodes[node]['status'] = 'failed'

            active_nodes = [n for n in graph.nodes if graph.nodes[n]['status'] == 'active']
            cascade_history.append(len(new_failures))
            step += 1

        # 计算最终损失
        total_failed = len([n for n in graph.nodes if graph.nodes[n]['status'] == 'failed'])
        return total_failed, graph

    def visualize(self, graph, title="Simulation Result"):
        # 可视化
        pos = nx.get_node_attributes(graph, 'pos')
        colors = ['red' if graph.nodes[n]['status'] == 'failed' else 'blue' for n in graph.nodes]

        plt.figure(figsize=(8, 6))
        nx.draw(graph, pos, node_color=colors, node_size=50, with_labels=False, edge_color="gray", alpha=0.6)
        plt.title(title)
        plt.show()

if __name__ == "__main__":
    # 设置保存路径
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，自动创建

    # 运行模拟
    print("正在初始化基础设施网络...")
    sim = CascadingSimulator(num_nodes=200, radius=0.12)  # 节点多一点

    print("模拟攻击...")
    sim.trigger_failure(num_initial_failures=3)

    print("计算级联...")
    final_loss, final_graph = sim.run_cascade()

    # 绘图自动保存
    print("正在绘图...")
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(final_graph, 'pos')
    colors = ['red' if final_graph.nodes[n]['status'] == 'failed' else 'blue' for n in final_graph.nodes]

    nx.draw(final_graph, pos, node_color=colors, node_size=50, edge_color="gray", alpha=0.4)
    plt.title(f"Simulation: {final_loss / 200:.1%} Failure Rate")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"cascade_{timestamp}.png")

    plt.savefig(save_path, dpi=300)
    print(f"图片已保存到: {save_path}")
    plt.show()