import re
import torch
from transformers import GenerationConfig
import pandas as pd

class TableRelevanceExtractor:
    """
    表格相关行提取器
    用于使用LLM识别表格中与问题相关的行
    """
    
    def __init__(self, model, tokenizer, device="cuda:0"):
        """
        初始化表格相关行提取器
        
        参数:
            model: 预训练语言模型
            tokenizer: 分词器
            device: 设备名称
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def get_relevant_rows(self, df, table_name, question, max_rows=15):
        """
        使用LLM来确定表格中与问题相关的行索引
        
        参数:
            df (pd.DataFrame): 表格数据
            table_name (str): 表格名称
            question (str): 问题文本
            max_rows (int): 返回的最大相关行索引数量
            
        返回:
            List[int]: 相关行索引列表，数量不超过max_rows
        """
        if df.empty:
            return []

        # --- 优化的提示模板 ---
        prompt = f"""Your task is to identify the most relevant row indices from the table below to answer the given question.

        Question: {question}

        Table Name: {table_name}
        Total Rows in Original Table: {len(df)}
        Table Schema (Columns): {df.columns.tolist()}

        Table Content Sample (up to 20 rows shown):
        {df.head(20).to_string(index=True)}

        Instructions:
        1. Analyze the Question and the Table Schema/Content.
        2. Identify rows containing information (direct or indirect) crucial for answering the question.
        3. If the table content sample seems incomplete (due to the 20-row limit), use the schema and sample to infer potential relevance in unseen rows, but prioritize rows shown.
        4. Return ONLY a comma-separated list of relevant row indices (integers). Example: 0,5,12,28
        5. If no rows seem relevant, return the exact text: No relevant rows

        Relevant row indices:"""

        # --- 生成配置 ---
        # 选项1: 更确定性的生成（通常更适合提取任务）
        simple_gen_config = GenerationConfig(
            max_new_tokens=150,  # 允许更多的token以确保安全
            do_sample=False,     # 使用贪婪解码或束搜索（如果模型支持）
            num_beams=3,         # 束搜索示例
            early_stopping=True, # 如果达到EOS则提前停止
            temperature=None,    # 对于do_sample=False不需要
            top_p=None           # 对于do_sample=False不需要
        )

        # --- 输入准备 ---
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"应用聊天模板时出错: {e}。回退到原始提示。")
            prompt_text = prompt # 模板失败时的回退方案

        # --- 输入标记化和长度计算 ---
        # 为输入标记计算更动态的max_length
        # 为生成配置的max_new_tokens和缓冲区预留空间
        model_max_len = getattr(self.model.config, 'max_position_embeddings', 4096) # 获取模型的最大长度
        reserved_space = simple_gen_config.max_new_tokens + 50 # 为输出和缓冲区预留空间
        input_max_len = model_max_len - reserved_space
        if input_max_len <= 0:
             print(f"警告: 模型最大长度({model_max_len})对于预留空间({reserved_space})来说太小。进行调整。")
             input_max_len = model_max_len // 2 # 或其他启发式方法

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,        # 如果有多个提示（这里没有），则填充到批次的最大长度
            truncation=True,     # 如果超过input_max_len则截断
            max_length=input_max_len # 使用计算的最大长度
        ).to(self.device)

        # 检查是否过度截断
        if inputs.input_ids.shape[1] >= input_max_len:
             print(f"警告: 表'{table_name}'的LLM过滤提示被截断至{input_max_len}个tokens。上下文可能丢失。")

        # --- LLM生成 ---
        relevant_indices = []
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=simple_gen_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # --- 输出解码和解析 ---
            # 仅解码生成的部分
            output_token_ids = outputs[0][inputs.input_ids.shape[1]:]
            output_text = self.tokenizer.decode(
                output_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            if "no relevant rows" in output_text.lower(): # 不区分大小写检查
                return []

            # 使用正则表达式提取索引，转换为int，验证是否在df.index中
            extracted_indices = set() # 使用集合处理LLM输出中的重复项
            for token in re.findall(r'\d+', output_text):
                try:
                    idx = int(token)
                    if idx in df.index: # 检查索引对于*原始*df是否有效
                        extracted_indices.add(idx)
                except ValueError:
                    continue # 忽略非整数标记

            # 将集合转回列表并排序以保持一致性（可选）
            relevant_indices = sorted(list(extracted_indices))

            # 限制为max_rows
            return relevant_indices[:max_rows]

        except Exception as e:
            print(f"警告: 表'{table_name}'的LLM相关性生成或解析过程中出错: {e}")
            return [] # 出错时返回空列表
    
    def get_relevant_columns(self, df, question_keywords, include_ids=True):
        """
        基于关键词匹配识别相关列
        可选择始终包含可能的ID/Name列
        
        参数:
            df (pd.DataFrame): 表格数据
            question_keywords (set): 问题中的小写关键词集合
            include_ids (bool): 是否始终包含潜在的ID/Name列
            
        返回:
            List[str]: 相关列名列表
        """
        relevant_cols = set()
        potential_id_cols = set()

        if df.empty:
            return []

        for col in df.columns:
            col_lower = str(col).lower() # 确保列名是字符串并且小写

            # 基本检查列是否看起来像ID/Name（启发式）
            if include_ids and ('id' in col_lower or 'key' in col_lower or 'name' in col_lower or 'identifier' in col_lower or 'code' in col_lower):
                 potential_id_cols.add(col)

            # 检查列名本身是否相关
            # 拆分列名，以防它是多词的（例如，"department_name"）
            col_name_parts = set(re.split(r'[_\s-]+', col_lower))
            if question_keywords.intersection(col_name_parts):
                 relevant_cols.add(col)

        # 合并相关列和潜在ID列
        final_relevant_cols = relevant_cols.union(potential_id_cols if include_ids else set())

        # 回退：如果没有识别到列，使用潜在ID或仅第一列
        if not final_relevant_cols:
            if potential_id_cols:
                 final_relevant_cols = potential_id_cols
            elif len(df.columns) > 0:
                 final_relevant_cols = {df.columns[0]} # 回退到第一列

        # 按原始顺序返回，确保至少有一列（如果可能）
        ordered_final_cols = [col for col in df.columns if col in final_relevant_cols]

        if not ordered_final_cols and len(df.columns) > 0:
             return [df.columns[0]] # 绝对回退：第一列
        elif not ordered_final_cols:
             return [] # df中没有列
        else:
             return ordered_final_cols