import pandas as pd
import json
import os
import re
from openai import OpenAI
from tqdm import tqdm

# =================配置区域=================
# 输入文件
DATA_FILE = "processed_policy_clusters.csv"  # 你的聚类结果
ENTITY_FILE = "Entity_Unified_Table.xlsx"  # 实体定义
RELATION_FILE = "Relationship_Unified_Table.xlsx"  # 关系定义
OUTPUT_FILE = "final_triples_for_graph.json"  # 输出结果

# API 配置 (请替换为你的 Key)
client = OpenAI(
    api_key="sk-849b31ef6a1f46e1b268ae4a9ae18b97",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# =========================================
class SchemaLoader:
    def __init__(self, entity_path, relation_path):
        self.entity_prompt = self._load_entities(entity_path)
        self.cluster_configs = self._load_relations(relation_path)

    def _read_csv_safe(self, path, **kwargs):
        """读取 CSV 的保底逻辑 (处理各种编码)"""
        encodings = ['gbk', 'utf-8', 'gb18030', 'utf-8-sig', 'ansi']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, on_bad_lines='skip', engine='python', **kwargs)
            except:
                continue
        raise ValueError(f"无法识别 CSV 编码: {path}")

    def _smart_load(self, path, **kwargs):
        """【核心修复】智能识别 Excel 或 CSV"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

        print(f"📂 正在读取: {path} ...")

        # 1. 尝试作为 Excel 读取
        if path.lower().endswith(('.xlsx', '.xls')):
            try:
                # Excel 读取不需要 encoding 参数
                return pd.read_excel(path, **kwargs)
            except Exception as e:
                print(f"⚠️ Excel 读取失败 ({e})，尝试作为 CSV 读取...")
                # 如果用户只是改了后缀名但内容还是csv，会走到这里

        # 2. 作为 CSV 读取
        return self._read_csv_safe(path, **kwargs)

    def _load_entities(self, path):
        """解析实体表"""
        df = self._smart_load(path)

        # 兼容中文列名和英文列名 (防止列名不匹配报错)
        # 寻找包含 "实体" 或 "Entity" 的列作为标准名
        col_map = {c: c for c in df.columns}
        canonical_col = next((c for c in df.columns if "标准" in c or "Canonical" in c), None)
        variants_col = next((c for c in df.columns if "变体" in c or "Variants" in c), None)
        type_col = next((c for c in df.columns if "类型" in c or "Type" in c), None)

        if not canonical_col:
            raise ValueError(f"实体表中找不到'标准实体名'列。现有列: {df.columns}")

        if type_col:
            df[type_col] = df[type_col].ffill()

        lines = []
        for _, row in df.iterrows():
            canon = row.get(canonical_col)
            vars_ = row.get(variants_col)

            if pd.isna(canon): continue

            if pd.notna(vars_):
                lines.append(f"- 遇到 '{vars_}' -> 统一为 '{canon}'")
            else:
                lines.append(f"- 标准实体: '{canon}'")
        return "\n".join(lines)

    def _load_relations(self, path):
        """解析关系表"""
        # 关系表通常没有标准表头，或者结构复杂，建议 header=None 读取原始数据
        # 但如果是 Excel，read_excel 默认第一行是表头。
        # 这里我们先按 header=None 读取，方便处理非结构化内容
        df = self._smart_load(path, header=None)

        configs = {}
        curr_clusters = []
        curr_desc = "提取关系"

        for _, row in df.iterrows():
            text = str(row[0])

            # 识别 Cluster 行 (例如: "A. 许可与禁止 (针对 Cluster 1, 5)")
            if "针对 Cluster" in text:
                curr_clusters = [int(x) for x in re.findall(r'\d+', text)]
                continue

            # 识别描述行
            if "用于回答" in text:
                curr_desc = text.replace("用于回答：", "").strip()
                continue

            # 跳过表头(包含"标准关系"字样) 和 空行
            if "标准关系" in text or pd.isna(row[0]) or pd.isna(row[4]):
                continue

            # 解析具体关系行
            if curr_clusters:
                # 假设列顺序: 0:Predicate, 3:Triggers, 4:Example
                # Excel读取后索引可能变化，这里做个简单防御
                try:
                    pred = row[0]
                    trig = row[3] if pd.notna(row[3]) else "无"
                    ex = row[4] if pd.notna(row[4]) else ""

                    rel_prompt = f"- 关系 '{pred}': 当文中暗示 ({trig}) 时使用。\n  参考示例: {ex}"

                    for cid in curr_clusters:
                        if cid not in configs:
                            configs[cid] = {"desc": curr_desc, "rels": []}
                        configs[cid]["rels"].append(rel_prompt)
                except IndexError:
                    continue  # 防止列数不够

        return configs

    def get_config(self, cluster_id):
        return self.cluster_configs.get(cluster_id, {
            "desc": "通用提取",
            "rels": ["- 提取文中核心实体间的关系"]
        })


def extract_triples(text, cluster_id, formulator, schema):
    config = schema.get_config(cluster_id)
    relation_str = "\n".join(config['rels'])

    # 调试：检查 Prompt 是否为空 (如果 Schema 加载失败，这里可能为空)
    if not schema.entity_prompt:
        print("⚠️ 警告: 实体 Prompt 为空，可能是 Entity 表读取失败！")

    prompt = f"""
    任务：你是一个AI政策领域的知识图谱专家。请基于【特定视角】从文本中抽取三元组。

    【核心视角】
    当前出版商(Publisher)是："{formulator}"
    你的目标是回答："{config['desc']}"

    【强制约束】
    1. **三元组的主语(Head)必须是出版商 "{formulator}"** (除非文中明确提到是第三方行为)。
    2. 严格遵守以下实体映射，不要创造新词：
    {schema.entity_prompt}

    【允许使用的关系 (Schema)】
    {relation_str}

    【待处理文本】
    "{text}"

    【输出要求】
    不要输出任何分析过程，只输出一个标准的 JSON 列表。
    格式：[{{"head": "{formulator}", "relation": "...", "tail": "..."}}]
    如果没有相关信息，输出 []。
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.01
        )
        content = completion.choices[0].message.content.strip()

        if "```" in content:
            if "json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            else:
                content = content.replace("```", "").strip()

        return json.loads(content)
    except Exception as e:
        # =========== 修改了这里 ===========
        print(f"\n❌ [Error] Cluster {cluster_id} 抽取失败: {e}")
        # 如果是认证失败，直接中断程序，不要跑完了才发现
        if "Authentication" in str(e) or "401" in str(e):
            raise e
        return []

def main():
    print("🚀 正在初始化...")
    try:
        schema = SchemaLoader(ENTITY_FILE, RELATION_FILE)
        print("✅ Schema 加载成功！")
    except Exception as e:
        print(f"❌ Schema 加载失败: {e}")
        # 这里打印详细错误栈，方便调试
        import traceback
        traceback.print_exc()
        return

    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到数据文件: {DATA_FILE}")
        return

    print(f"📂 正在加载数据文件: {DATA_FILE} ...")
    try:
        df = pd.read_csv(DATA_FILE, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(DATA_FILE, encoding='gbk')
        except:
            print("❌ 数据文件编码无法识别，请转为 UTF-8")
            return

    print(f"✅ 载入数据: {len(df)} 条。开始抽取...\n")
    print("-" * 60)

    all_triples = []

    # 限制测试前10条，正式跑的时候去掉 .head(10)
    # for index, row in tqdm(df.head(10).iterrows(), total=10, desc="抽取进度"):
    for index, row in tqdm(df.iterrows(), total=len(df), desc="抽取进度"):
        triples = extract_triples(
            text=row['original_sentence'],
            cluster_id=row['cluster_label'],
            formulator=row['formulator'],
            schema=schema
        )

        if triples:
            tqdm.write(f"🎯 [Cluster {row['cluster_label']}] {row['formulator']}:")
            for t in triples:
                tqdm.write(f"   ({t['head']}) --[{t['relation']}]--> ({t['tail']})")

                t['cluster_id'] = int(row['cluster_label'])
                t['source_text'] = row['original_sentence']
                all_triples.append(t)
            tqdm.write("")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)
    print("-" * 60)
    print(f"🎉 全部完成！共提取 {len(all_triples)} 个三元组。")
    print(f"💾 结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()