import torch
import faiss
import networkx as nx
import json
import numpy as np
from openai import OpenAI

# 导入你之前定义的图模型 (假设文件名为 graph_bert.py)
from graph_bert import SciBERT_Graph_MultiAttention

# 配置 API
client = OpenAI(
    api_key="sk-849b31ef6a1f46e1b268ae4a9ae18b97",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


class PolicyGraphRAG:
    def __init__(self, triples_file, original_texts_file):
        self.triples_file = triples_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. 加载模型
        self.model = SciBERT_Graph_MultiAttention().to(self.device)
        self.model.eval()

        # 2. 初始化图结构 (用 NetworkX 模拟 Neo4j)
        self.graph = nx.MultiDiGraph()
        self.load_graph_data()

        # 3. 初始化向量索引 (Faiss)
        self.index = faiss.IndexFlatL2(768)  # 768是SciBERT维度
        self.stored_nodes = []  # 记录索引对应的节点信息
        self.build_vector_index()

    def load_graph_data(self):
        """加载三元组构建图"""
        with open(self.triples_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)

        for t in triples:
            # 添加边：Head -> Tail，属性包含 relation 和 原始文本
            self.graph.add_edge(
                t['head'],
                t['tail'],
                relation=t['relation'],
                source_text=t.get('source_text', '')
            )
            # 同时也给节点加属性，方便后续检索
            if 'formulator' in t:
                self.graph.nodes[t['head']]['type'] = 'Publisher'

    def build_vector_index(self):
        """
        核心步骤：利用你的 Attention 模型生成增强向量并存入库
        注意：这里我们把'原始文本'作为索引单位
        """
        print("正在构建图增强向量索引...")

        # 收集图中所有的边作为“文档片段”
        vectors = []

        for u, v, data in self.graph.edges(data=True):
            text = data.get('source_text', '')
            if not text: continue

            # 准备模型的输入：文本 + 节点 + 边
            # 这里的 nodes 和 edges 是从当前三元组中提取的
            nodes = [u, v]
            edges = [data['relation']]

            # 调用你的模型生成向量
            with torch.no_grad():
                # 构造符合模型输入的格式
                tokenizer = self.model.tokenizer
                inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)

                # 核心：使用你的 Multi-Attention 模型
                vec = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    node_texts_batch=[nodes],
                    edge_texts_batch=[edges]
                )

            vectors.append(vec.cpu().numpy())
            self.stored_nodes.append({
                "head": u, "tail": v, "relation": data['relation'], "text": text
            })

        # 存入 Faiss
        if vectors:
            vectors_np = np.vstack(vectors)
            self.index.add(vectors_np)
            print(f"索引构建完成，共 {len(vectors)} 条知识条目。")

    def retrieve(self, query, top_k=3):
        """
        检索阶段：
        1. 用 SciBERT 编码 Query (此时没有图结构，只用文本路)
        2. 在 Faiss 中找最相似的 Policy
        3. 在 Graph 中找这些 Policy 的邻居 (扩展上下文)
        """
        # 1. 编码 Query
        # 注意：Query 没有图结构，所以 node_texts 和 edge_texts 传空
        with torch.no_grad():
            tokenizer = self.model.tokenizer
            inputs = tokenizer([query], return_tensors="pt", padding=True).to(self.device)
            query_vec = self.model(
                inputs['input_ids'],
                inputs['attention_mask'],
                [[]], [[]]  # 空图
            ).cpu().numpy()

        # 2. 向量搜索
        D, I = self.index.search(query_vec, top_k)

        retrieved_knowledge = []
        seen_triples = set()

        for idx in I[0]:
            if idx == -1: continue
            item = self.stored_nodes[idx]

            # 添加命中的直接知识
            knowledge_str = f"{item['head']} --[{item['relation']}]--> {item['tail']}"
            retrieved_knowledge.append(f"[直接相关] {knowledge_str}\n(原文: {item['text']})")
            seen_triples.add((item['head'], item['relation'], item['tail']))

            # 3. 图谱扩展 (Graph Traversal) - 关键步骤！
            # 既然找到了这个节点，看看它周围还有什么？
            # 例如：查到了 Elsevier 禁止 AI 作者，顺便看看 Elsevier 允不允许润色
            center_node = item['head']
            neighbors = list(self.graph.out_edges(center_node, data=True))

            for u, v, data in neighbors[:2]:  # 限制扩展数量防止噪音
                if (u, data['relation'], v) not in seen_triples:
                    ext_str = f"{u} --[{data['relation']}]--> {v}"
                    retrieved_knowledge.append(f"[图谱联想] {ext_str}")
                    seen_triples.add((u, data['relation'], v))

        return "\n\n".join(retrieved_knowledge)

    def ask(self, query):
        """生成回答"""
        context = self.retrieve(query)

        prompt = f"""
        你是一个基于知识图谱的 AI 政策专家。请根据以下检索到的知识回答用户问题。

        【检索到的知识图谱路径】
        {context}

        【用户问题】
        {query}

        【回答要求】
        1. 必须基于提供的知识回答，并在回答中标注来源（如 Elsevier, IEEE）。
        2. 如果涉及不同出版商的对比，请列出差异。
        3. 如果知识中没有提到，请直接说不知道。
        """

        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content


# ================= 使用示例 =================
if __name__ == "__main__":
    # 初始化系统
    rag_system = PolicyGraphRAG(
        triples_file="final_triples_for_graph.json",
        original_texts_file="processed_policy_clusters.csv"  # 这里其实暂时没用到，因为三元组里有 source_text
    )

    # 提问测试
    question = "Elsevier 和 IEEE 对于 AI 署名的规定有什么不同？"
    print(f"用户提问: {question}")

    answer = rag_system.ask(question)
    print("\nGraphRAG 回答:\n", answer)