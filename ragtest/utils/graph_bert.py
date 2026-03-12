import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SciBERT_Graph_MultiAttention(nn.Module):
    def __init__(self, model_name='allenai/scibert_scivocab_uncased', hidden_dim=768, num_heads=8):
        super(SciBERT_Graph_MultiAttention, self).__init__()
        print(f"正在加载图增强模型: {model_name} ...")

        # 1. 共享的基础编码器 (SciBERT)
        # 无论是文本、节点还是边，最初都由 SciBERT 进行编码，保证语义空间一致
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 2. 原有的文本注意力层 (保留你的前期工作)
        self.text_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 3. 新增：节点间注意力 (Point-to-Point Attention)
        # 用于捕捉实体(Entities)之间的关系
        self.node_mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.node_norm = nn.LayerNorm(hidden_dim)  # Add & Norm

        # 4. 新增：边间注意力 (Edge-to-Edge Attention)
        # 用于捕捉关系(Relations)之间的逻辑
        self.edge_mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.edge_norm = nn.LayerNorm(hidden_dim)  # Add & Norm

        # 5. 最终融合层 (Fusion Layer)
        # 将 文本向量 + 节点向量 + 边向量 融合
        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)

    def get_text_embedding(self, input_ids, attention_mask):
        """计算文本的加权向量 (你原来的逻辑)"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_state = outputs.last_hidden_state

        # Text Attention
        attn_scores = self.text_attention(h_state)  # (batch, seq, 1)
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        text_vec = torch.sum(h_state * attn_weights, dim=1)  # (batch, dim)
        return text_vec

    def get_graph_component_embedding(self, text_list, device):
        """辅助函数：将实体或关系的文本列表转为向量"""
        if not text_list:  # 如果没有实体/关系，返回零向量
            return None

        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        # 使用 [CLS] 作为该实体/关系的初始表示
        return outputs.last_hidden_state[:, 0, :].unsqueeze(0)  # (1, num_items, dim)

    def forward(self, input_ids, attention_mask, node_texts_batch, edge_texts_batch):
        """
        input_ids: 原始句子的token ids
        node_texts_batch: 一个list，包含该句子对应的所有实体文本 ['AI', 'Author', ...]
        edge_texts_batch: 一个list，包含该句子对应的所有关系文本 ['prohibits', 'requires', ...]
        """
        device = input_ids.device
        batch_size = input_ids.size(0)

        # ----------------------------------------
        # 1. 获取文本向量 (Text Representation)
        # ----------------------------------------
        text_vec = self.get_text_embedding(input_ids, attention_mask)  # (batch, dim)

        # ----------------------------------------
        # 2. 获取节点向量并应用 Attention (Node Representation)
        # ----------------------------------------
        # 注意：这里简化处理，假设 batch=1 或者需要自行padding节点数量
        # 实际训练中，需要把不同句子的节点pad到相同长度

        node_vec_final = torch.zeros_like(text_vec)

        # 对 batch 中的每个样本单独处理图结构 (因为节点数不同)
        for i in range(batch_size):
            # 2.1 编码节点
            nodes = node_texts_batch[i]  # list of strings
            if nodes:
                node_emb = self.get_graph_component_embedding(nodes, device)  # (1, num_nodes, dim)

                # 2.2 Point-to-Point Attention ! (核心修改点)
                # Query=Key=Value=Nodes (Self-Attention)
                node_out, _ = self.node_mha(node_emb, node_emb, node_emb)

                # Residual connection & Norm
                node_out = self.node_norm(node_emb + node_out)

                # Pooling: 将所有节点向量聚合为一个向量 (例如 Mean Pooling)
                node_vec_final[i] = torch.mean(node_out, dim=1)
            else:
                # 如果没有提取出节点，可以用文本向量代替，或者保持为0
                node_vec_final[i] = text_vec[i]

        # ----------------------------------------
        # 3. 获取边向量并应用 Attention (Edge Representation)
        # ----------------------------------------
        edge_vec_final = torch.zeros_like(text_vec)

        for i in range(batch_size):
            # 3.1 编码边
            edges = edge_texts_batch[i]
            if edges:
                edge_emb = self.get_graph_component_embedding(edges, device)  # (1, num_edges, dim)

                # 3.2 Edge-to-Edge Attention ! (核心修改点)
                edge_out, _ = self.edge_mha(edge_emb, edge_emb, edge_emb)
                edge_out = self.edge_norm(edge_emb + edge_out)

                # Pooling
                edge_vec_final[i] = torch.mean(edge_out, dim=1)
            else:
                edge_vec_final[i] = text_vec[i]

        # ----------------------------------------
        # 4. 多重特征融合 (Multi-View Fusion)
        # ----------------------------------------
        # 拼接: [Text, Nodes, Edges] -> (batch, dim*3)
        combined = torch.cat([text_vec, node_vec_final, edge_vec_final], dim=1)

        # 投影回 hidden_dim
        final_embedding = self.fusion_layer(combined)

        return final_embedding