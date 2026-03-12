import json
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
import random

# ================= 配置区域 =================
# 你的三元组文件 (由 extraction 脚本生成)
TRIPLES_FILE = "final_triples_for_graph.json"

# 你的原始数据文件 (包含 original_sentence 和 cluster_label)
CLUSTERS_FILE = "processed_policy_clusters.csv"

# 结果输出目录
OUTPUT_DIR = "evaluation_results"

# 画图时的字体设置 (防止中文乱码，Windows通常是SimHei，Mac可能是Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ===========================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class KGEvaluator:
    def __init__(self):
        print("🚀 正在初始化评估器...")

        # 1. 加载三元组
        if not os.path.exists(TRIPLES_FILE):
            raise FileNotFoundError(f"找不到三元组文件: {TRIPLES_FILE}")
        with open(TRIPLES_FILE, 'r', encoding='utf-8') as f:
            self.triples = json.load(f)
        print(f"✅ 加载三元组: {len(self.triples)} 条")

        # 2. 加载聚类原始数据
        if not os.path.exists(CLUSTERS_FILE):
            print(f"⚠️ 警告: 找不到聚类文件 {CLUSTERS_FILE}，表示学习评估可能无法运行。")
            self.df_clusters = pd.DataFrame()
        else:
            try:
                self.df_clusters = pd.read_csv(CLUSTERS_FILE)
                print(f"✅ 加载原始数据: {len(self.df_clusters)} 条")
            except:
                self.df_clusters = pd.read_csv(CLUSTERS_FILE, encoding='gbk')

        # 3. 构建图结构
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 NetworkX 图对象"""
        G = nx.MultiDiGraph()  # 使用多重有向图 (允许两点间多条边)
        for t in self.triples:
            if t['head'] and t['tail']:
                G.add_edge(t['head'], t['tail'], relation=t['relation'])
        return G

    # ================= [模块1] 拓扑结构评估 =================
    def eval_topology(self):
        print("\n📊 --- [1] 图谱拓扑结构评估 ---")
        G = self.graph

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        if num_nodes == 0:
            print("❌ 图为空，跳过拓扑评估。")
            return

        density = nx.density(G)

        # 连通性 (转为无向图计算)
        undirected_G = G.to_undirected()
        if nx.is_connected(undirected_G):
            components = 1
            max_component_ratio = 1.0
        else:
            components = nx.number_connected_components(undirected_G)
            largest_cc = max(nx.connected_components(undirected_G), key=len)
            max_component_ratio = len(largest_cc) / num_nodes

        # 平均度
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0

        stats = {
            "节点总数 (Nodes)": num_nodes,
            "边总数 (Edges)": num_edges,
            "网络密度 (Density)": f"{density:.5f}",
            "平均度 (Avg Degree)": f"{avg_degree:.2f}",
            "连通分量数 (Components)": components,
            "最大连通子图占比": f"{max_component_ratio:.2%}"
        }

        print("-" * 30)
        for k, v in stats.items():
            print(f"{k:<20}: {v}")
        print("-" * 30)

        # 保存到文本文件
        with open(f"{OUTPUT_DIR}/topology_stats.txt", "w", encoding="utf-8") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

    # ================= [模块2] 逻辑一致性评估 =================
    def eval_logic_consistency(self):
        print("\n🔍 --- [2] 逻辑一致性检查 (冲突检测) ---")
        conflicts = []

        # 定义互斥词对 (可以根据实际情况补充)
        conflict_pairs = [
            ("prohibit", "allow"),
            ("restrict", "permit"),
            ("not accept", "welcome"),
            ("denies", "accepts"),
            ("does not require", "requires"),
            ("prohibited", "encouraged")
        ]

        # 遍历所有节点对
        for u in self.graph.nodes():
            for v in self.graph.neighbors(u):
                # 获取两点间的所有边
                edges = self.graph.get_edge_data(u, v)
                relations = [edges[key]['relation'].lower() for key in edges]

                # 如果两点间只有一条关系，自然没冲突
                if len(relations) < 2: continue

                # 两两比对
                checked = set()
                for r1 in relations:
                    for r2 in relations:
                        if r1 == r2 or (r1, r2) in checked or (r2, r1) in checked:
                            continue

                        checked.add((r1, r2))
                        for neg, pos in conflict_pairs:
                            # 检查是否同时包含互斥词
                            if (neg in r1 and pos in r2) or (neg in r2 and pos in r1):
                                conflicts.append({
                                    "Publisher": u,
                                    "Object": v,
                                    "Relation A": r1,
                                    "Relation B": r2,
                                    "Conflict Type": f"{neg} vs {pos}"
                                })

        print(f"👉 检测到潜在逻辑冲突: {len(conflicts)} 处")

        if len(conflicts) > 0:
            print("⚠️ 示例冲突:")
            print(conflicts[0])
            df_conflicts = pd.DataFrame(conflicts)
            save_path = f"{OUTPUT_DIR}/logic_conflicts.csv"
            df_conflicts.to_csv(save_path, index=False)
            print(f"详细冲突列表已保存至: {save_path}")
        else:
            print("✅ 未检测到明显的逻辑冲突。Schema 约束效果良好！")

    # ================= [模块3] 表示学习质量评估 =================
    def eval_representation_quality(self):
        print("\n🎨 --- [3] 表示学习质量评估 (可视化) ---")

        if self.df_clusters.empty or 'cluster_label' not in self.df_clusters.columns:
            print("❌ 缺少聚类标签数据，跳过可视化。")
            return

        # 过滤掉无效数据
        df = self.df_clusters.dropna(subset=['original_sentence', 'cluster_label'])

        # 为了演示速度，如果数据量太大，随机采样 200 条
        if len(df) > 300:
            print("数据量较大，随机采样 300 条进行可视化...")
            df = df.sample(n=300, random_state=42)

        labels = df['cluster_label'].values
        texts = df['original_sentence'].tolist()

        # 1. 计算 Baseline 向量 (原始 SciBERT)
        print("⏳ 正在计算 Baseline (SciBERT) 向量...")
        baseline_vecs = self._get_scibert_embeddings(texts)

        # 2. 计算评估指标 (Baseline)
        sil_score = silhouette_score(baseline_vecs, labels)
        print(f"📌 [Baseline] 轮廓系数 (Silhouette Score): {sil_score:.4f}")

        # 3. 绘制 Baseline t-SNE
        self._plot_tsne(baseline_vecs, labels, "t-SNE Visualization (Baseline SciBERT)")

        print("\n💡 提示: 如果你已经跑出了 'Graph-Enhanced' 向量，")
        print("   请调用 _plot_tsne(custom_vectors, labels, 'Graph Enhanced') 进行对比。")

    def _get_scibert_embeddings(self, texts):
        """使用 HuggingFace 加载 SciBERT 提取 [CLS] 向量"""
        # 如果下载慢，可以换成本地路径或 'bert-base-uncased' 测试
        model_name = 'allenai/scibert_scivocab_uncased'
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        except:
            print(f"⚠️ 无法连接 HuggingFace，尝试使用 'bert-base-uncased' 代替...")
            model_name = 'bert-base-uncased'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

        vecs = []
        model.eval()

        # 批处理防止内存溢出
        batch_size = 8
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # 取 [CLS] token (Batch, Hidden)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            vecs.append(embeddings)

        return np.vstack(vecs)

    def _plot_tsne(self, X, labels, title):
        """绘制并保存 t-SNE 图"""
        print(f"正在绘制 {title} ...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            x=X_2d[:, 0], y=X_2d[:, 1],
            hue=labels,
            palette="tab10",  # 颜色盘
            style=labels,  # 不同形状
            s=80, alpha=0.8
        )
        plt.title(title, fontsize=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()

        save_path = f"{OUTPUT_DIR}/{title.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        print(f"🖼️ 图片已保存至: {save_path}")
        # plt.show() # 如果在服务器运行，请注释掉此行

    # ================= [模块4] 人工评估采样 =================
    def prepare_human_eval_set(self, sample_size=50):
        print("\n📝 --- [4] 生成人工评估数据集 ---")

        if self.df_clusters.empty:
            print("❌ 没有原始文本数据，无法生成评估集。")
            return

        # 随机抽取
        real_size = min(len(self.df_clusters), sample_size)
        sampled_df = self.df_clusters.sample(n=real_size, random_state=42)

        eval_data = []
        for idx, row in sampled_df.iterrows():
            text = row.get('original_sentence', '')
            # 在三元组结果中查找该文本对应的提取结果
            # 注意：这里做简单的文本匹配，如果你的json里存了cluster_id会更准
            extracted_triples = []
            for t in self.triples:
                # 简单清洗对比，防止空格差异
                if t.get('source_text', '').strip() == text.strip():
                    extracted_triples.append(f"({t['head']} -> {t['relation']} -> {t['tail']})")

            eval_data.append({
                "Sample ID": idx,
                "Cluster": row.get('cluster_label', 'Unknown'),
                "Original Text": text,
                "Model Output": " | ".join(extracted_triples) if extracted_triples else "No Triple Extracted",
                "Is Correct? (1/0)": "",  # 人工填写
                "Is Missing Info? (1/0)": "",  # 人工填写
                "Error Type": ""  # 人工填写 (幻觉/关系错/实体错)
            })

        df_eval = pd.DataFrame(eval_data)
        save_path = f"{OUTPUT_DIR}/human_eval_sample.xlsx"
        df_eval.to_excel(save_path, index=False)
        print(f"✅ 已生成 Excel 表格: {save_path}")
        print("👉 请打开文件，手动填写 'Is Correct?' 和 'Is Missing Info?' 列。")


if __name__ == "__main__":
    print("=== 开始 KG 效果评估 ===")

    # 初始化
    try:
        evaluator = KGEvaluator()

        # 1. 跑拓扑分析
        evaluator.eval_topology()

        # 2. 跑逻辑检查
        evaluator.eval_logic_consistency()

        # 3. 跑可视化 (Baseline)
        evaluator.eval_representation_quality()

        # 4. 生成人工打分表
        evaluator.prepare_human_eval_set(sample_size=50)

        print("\n=== 评估完成！请查看 evaluation_results 文件夹 ===")

    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()