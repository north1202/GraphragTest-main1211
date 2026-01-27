import requests
import json

# 请确保您的 GraphRAG 服务端已启动，并且端口号（此处为 8012）正确
url = "http://localhost:8012/v1/chat/completions"
headers = {"Content-Type": "application/json"}


def send_query(mode, question):
    """
    封装发送请求的函数，方便切换不同模式
    :param mode: 搜索模式 (global, local, full)
    :param question: 您想问的问题
    """
    # 根据您的后端配置，模型名称可能不同，这里保留您原本的命名习惯
    model_map = {
        "global": "graphrag-global-search:latest",
        "local": "graphrag-local-search:latest",
        "full": "full-model:latest"  # 如果您的后端支持混合搜索
    }

    if mode not in model_map:
        print(f"模式 {mode} 不支持")
        return

    payload = {
        "model": model_map[mode],
        "messages": [
            {
                "role": "system",
                "content": "你是一个学术出版合规助手。请基于知识图谱中的数据回答用户关于投稿规则的问题。"
            },
            {
                "role": "user",
                "content": f"{question} 请用中文回答。"
            }
        ],
        "temperature": 0.1,  # 对于规则查询，建议调低温度以保证准确性，减少幻觉
        "stream": False
    }

    print(f"\n[{mode.upper()} SEARCH] 正在提问: {question}")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            print(f"--- 回答 ---\n{content}\n----------------")
        else:
            print(f"请求失败: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"发生错误: {e}")


# ==========================================
# 针对您的出版规则数据的测试场景
# ==========================================

# 场景 1: 全局搜索 (Global Search)
# 适用：宏观总结、跨出版社对比。GraphRAG 会利用社区摘要（Community Summaries）来回答。
# 您的数据场景：询问整体趋势、所有出版商的共性。
query_global = "目前主流出版社对于AI署名（AI Authorship）的普遍态度是什么？有哪些主要的共性规则？"

# 场景 2: 本地搜索 (Local Search)
# 适用：精确查找实体细节。GraphRAG 会在图谱中定位具体节点（如 Elsevier 节点）并检索其邻居关系。
# 您的数据场景：查询特定期刊、特定规则的具体条款。
query_local = "Elsevier 和 ASHA 对于图片生成（AI Image Generation）的具体规定分别是什么？它们允许润色吗？"

# 场景 3: 混合/全量搜索 (Full Search) - 如果支持
# 适用：既需要细节也需要宏观背景。
query_full = "如果我是一名审稿人，我可以使用ChatGPT来辅助审稿吗？请列出明确禁止此行为的机构。"

# ==========================================
# 执行测试 (取消注释以运行)
# ==========================================

if __name__ == "__main__":
    # 1. 测试全局概览能力
    send_query("global", query_global)

    # 2. 测试细节查证能力 (这是您最需要的)
    send_query("local", query_local)

    # 3. 测试混合能力
    send_query("full", query_full)