from neo4j import GraphDatabase
import json
import os

# ================= 配置区域 =================
# Neo4j 连接配置 (请修改为你的实际密码)
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "20021202")  # <--- 修改这里的密码

# 输入数据文件
INPUT_JSON = "final_triples_for_graph.json"


# ===========================================

class KnowledgeGraphBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def clear_database(self):
        """清空数据库 (慎用，每次重跑建议先清空)"""
        print("正在清空数据库...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("数据库已清空。")

    def create_constraints(self):
        """创建唯一性约束，防止重复节点，提高查询速度"""
        print("正在创建索引和约束...")
        with self.driver.session() as session:
            # 针对 Publisher 建立约束
            try:
                session.run("CREATE CONSTRAINT FOR (p:Publisher) REQUIRE p.name IS UNIQUE")
            except Exception:
                pass  # 约束可能已存在

            # 针对 Concept (客体实体) 建立约束
            try:
                session.run("CREATE CONSTRAINT FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            except Exception:
                pass

    def import_data(self, json_path):
        if not os.path.exists(json_path):
            print(f"错误：找不到文件 {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"开始导入 {len(data)} 条三元组...")

        with self.driver.session() as session:
            for i, item in enumerate(data):
                head = item.get('head')
                tail = item.get('tail')
                relation = item.get('relation')
                source_text = item.get('source_text', '')
                cluster_id = item.get('cluster_id', -1)

                # 数据清洗：去除空值或特殊字符
                if not head or not tail or not relation:
                    continue

                # 清洗关系名：Neo4j关系类型不能包含空格，建议转为下划线大写
                # 例如: "prohibits use of" -> "PROHIBITS_USE_OF"
                clean_relation = relation.replace(" ", "_").upper().replace("-", "_")

                # 执行写入
                session.execute_write(
                    self._create_path,
                    head, clean_relation, tail, source_text, cluster_id
                )

                if (i + 1) % 50 == 0:
                    print(f"已处理 {i + 1} 条...")

        print("导入完成！")

    @staticmethod
    def _create_path(tx, head_name, relation_type, tail_name, text, cluster_id):
        """
        Cypher 语句逻辑：
        1. MERGE Publisher (如果不存在就创建，存在就复用)
        2. MERGE Concept (如果不存在就创建)
        3. MERGE Relationship (建立带属性的边)
        """
        # 注意：关系类型 (relation_type) 不能直接参数化，需要使用 f-string 注入
        # 但我们之前做了清洗，相对安全
        query = f"""
        MERGE (h:Publisher {{name: $head_name}})
        MERGE (t:Concept {{name: $tail_name}})
        MERGE (h)-[r:`{relation_type}`]->(t)
        SET r.source_text = $text,
            r.cluster_id = $cluster_id,
            r.original_relation = $orig_rel
        """
        tx.run(query,
               head_name=head_name,
               tail_name=tail_name,
               text=text,
               cluster_id=cluster_id,
               orig_rel=relation_type)  # 保留一个原始关系名备查


if __name__ == "__main__":
    # 1. 确保 JSON 文件在当前目录或指定目录
    # 如果文件在上一级，请修改 INPUT_JSON 路径

    kg = KnowledgeGraphBuilder(URI, AUTH)
    try:
        kg.clear_database()  # 1. 清库
        kg.create_constraints()  # 2. 建索引
        kg.import_data(INPUT_JSON)  # 3. 导数据
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        kg.close()