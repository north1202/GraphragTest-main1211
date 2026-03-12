import json
import re
import pandas as pd
from tqdm import tqdm
import os
from openai import OpenAI

from graph_rag_qa import PolicyGraphRAG
# ================= 配置环境参数 =================

API_KEY = "sk-849b31ef6a1f46e1b268ae4a9ae18b97"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 以通义千问兼容模式为例
MODEL_NAME = "qwen-plus"
JUDGE_MODEL = "qwen-max"  #

BENCHMARK_PATH = "qa_benchmark.json"
OUTPUT_REPORT_PATH = "llm_evaluation_report.csv"
# ===============================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def __init__(self, benchmark_path):
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        self.benchmark_data = json.load(f)

    # 实例化真实的系统
    print("正在初始化真实的 GraphRAG 系统，这可能需要加载向量索引和图谱...")
    self.real_rag_system = PolicyGraphRAG()

    # 确保在此调用加载数据的方法，例如：
    # self.real_rag_system.load_graph_data()
    # self.real_rag_system.load_vector_index()

    def _call_llm(self, prompt, model=MODEL_NAME):
        """调用大语言模型进行文本生成。"""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # 采用低温度以保证输出的确定性
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM API 调用失败: {e}")
            return "Error generating response."

    def _mock_naive_retrieve(self, query):
        """执行真实的纯向量检索 (Naive RAG)"""
        # 假设 retrieve_vector_only 是您系统中仅依靠 FAISS 进行检索的函数
        # 如果没有，您需要在 PolicyGraphRAG 中添加一个仅查 FAISS 不查 Graph 的方法
        context_nodes = self.real_rag_system.retrieve_vector_only(query, top_k=3)
        return "\n".join([str(node) for node in context_nodes])

    def _mock_graph_retrieve(self, query):
        """执行真实的图增强检索 (GraphRAG)"""
        # 假设 retrieve_graph_enhanced 是结合了向量与图谱多跳扩展的检索函数
        context_triples = self.real_rag_system.retrieve_graph_enhanced(query, top_k=3)
        return "\n".join([str(triple) for triple in context_triples])

    def calculate_hit_rate(self, retrieved_context, required_entities):
        """
        量化检索效能：计算所需实体在检索上下文中的命中率 (Hit Rate)。
        """
        if not required_entities:
            return 1.0

        context_lower = retrieved_context.lower()
        hits = sum(1 for entity in required_entities if entity.lower() in context_lower)
        return hits / len(required_entities)

    def llm_as_a_judge(self, question, ground_truth, generated_answer):
        """
        利用 LLM-as-a-Judge 范式对生成答案的精确度 (Precision) 进行盲评量化。
        评分维度：事实一致性与完整性 (1-5分)。
        """
        judge_prompt = f"""
        作为一名客观的学术评估专家，请根据提供的标准答案评估模型生成的答案。

        [评估标准]
        - 1分：完全不相关或存在严重事实错误（幻觉）。
        - 3分：包含部分正确信息，但遗漏了关键细节或包含轻微不准确之处。
        - 5分：极其准确、完整，并且与标准答案的事实完全一致。

        [数据]
        问题: {question}
        标准答案: {ground_truth}
        模型生成的答案: {generated_answer}

        请仔细对比，并仅输出一个 1 到 5 之间的整数作为最终得分，不要输出任何解释说明。
        """

        score_str = self._call_llm(judge_prompt, model=JUDGE_MODEL)

        # 提取整数评分
        match = re.search(r'\d+', score_str)
        if match:
            score = int(match.group())
            return max(1, min(score, 5))  # 确保分数在 1-5 之间
        return 1  # 解析失败默认最低分

    def run_evaluation(self):
        """执行全流程对照实验。"""
        results = []

        print(f"开始执行自动化评测，共 {len(self.benchmark_data)} 个测试用例...")

        for item in tqdm(self.benchmark_data):
            query = item["question"]
            gt_answer = item["ground_truth_answer"]
            entities = item.get("required_entities", [])

            # --------------------------------------------------
            # 实验组 1: Pure LLM (Baseline)
            # --------------------------------------------------
            prompt_pure = f"请准确回答以下问题：\n{query}"
            ans_pure = self._call_llm(prompt_pure)
            score_pure = self.llm_as_a_judge(query, gt_answer, ans_pure)

            # --------------------------------------------------
            # 实验组 2: Naive RAG (传统检索增强)
            # --------------------------------------------------
            ctx_naive = self._mock_naive_retrieve(query)
            hr_naive = self.calculate_hit_rate(ctx_naive, entities)
            prompt_naive = f"基于以下参考信息回答问题。\n参考信息：{ctx_naive}\n问题：{query}"
            ans_naive = self._call_llm(prompt_naive)
            score_naive = self.llm_as_a_judge(query, gt_answer, ans_naive)

            # --------------------------------------------------
            # 实验组 3: GraphRAG (图谱增强)
            # --------------------------------------------------
            ctx_graph = self._mock_graph_retrieve(query)
            hr_graph = self.calculate_hit_rate(ctx_graph, entities)
            prompt_graph = f"基于以下包含图结构知识的参考信息回答问题。\n参考信息：{ctx_graph}\n问题：{query}"
            ans_graph = self._call_llm(prompt_graph)
            score_graph = self.llm_as_a_judge(query, gt_answer, ans_graph)

            # 记录本轮测试结果
            results.append({
                "Query ID": item.get("query_id", "Unknown"),
                "Question": query,
                "Hit Rate (Naive)": hr_naive,
                "Hit Rate (Graph)": hr_graph,
                "Precision Score (Pure LLM)": score_pure,
                "Precision Score (Naive RAG)": score_naive,
                "Precision Score (GraphRAG)": score_graph
            })

        # 汇总与导出报告
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_REPORT_PATH, index=False, encoding='utf-8-sig')

        print("\n=== 实验评测结果汇总 ===")
        print(f"平均检索命中率 - Naive RAG: {df['Hit Rate (Naive)'].mean():.2%}")
        print(f"平均检索命中率 - GraphRAG: {df['Hit Rate (Graph)'].mean():.2%}")
        print(f"平均生成精准度 (5分制) - Pure LLM: {df['Precision Score (Pure LLM)'].mean():.2f}")
        print(f"平均生成精准度 (5分制) - Naive RAG: {df['Precision Score (Naive RAG)'].mean():.2f}")
        print(f"平均生成精准度 (5分制) - GraphRAG: {df['Precision Score (GraphRAG)'].mean():.2f}")
        print(f"详细报告已保存至: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    # 确保基准文件存在后执行
    if os.path.exists(BENCHMARK_PATH):
        evaluator = LlmRagEvaluator(BENCHMARK_PATH)
        evaluator.run_evaluation()
    else:
        print(f"请先创建评测基准数据文件：{BENCHMARK_PATH}")