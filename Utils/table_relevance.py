import re
import torch
from transformers import GenerationConfig
import pandas as pd

class TableRelevanceExtractor:
    """
    表格相关行提取器
    用于使用LLM识别表格中与问题相关的行
    """
    
    def __init__(self, llm, tokenizer, device="cuda:0"):
        """
        初始化表格相关行提取器
        
        参数:
            llm: VLLM模型实例
            tokenizer: 分词器
            device: 设备名称
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = device
    
    def extract_relevant_rows(self, table_content, question, use_llm=False):
        """
        提取与问题相关的表格行
        
        参数:
            table_content: 表格内容
            question: 问题
            use_llm: 是否使用LLM进行提取
            
        返回:
            处理后的表格内容
        """
        if not use_llm:
            return table_content
            
        # 解析表格内容
        tables = self._parse_tables(table_content)
        if not tables:
            return table_content
            
        # 处理每个表格
        processed_tables = []
        for table in tables:
            df = table.get('df')
            if df is None or df.empty:
                processed_tables.append(table['raw_content'])
                continue
                
            # 获取相关行
            relevant_indices = self.get_relevant_rows(df, table.get('name', 'Unknown'), question)
            if not relevant_indices or relevant_indices == ["No relevant rows"]:
                processed_tables.append(table['raw_content'])
                continue
                
            # 筛选相关行
            try:
                indices = [int(idx) for idx in relevant_indices if idx.isdigit()]
                if indices:
                    filtered_df = df.iloc[indices]
                    # 重新格式化为表格
                    if 'markdown' in table:
                        processed_content = self._format_as_markdown(filtered_df, table.get('name', ''))
                    else:
                        processed_content = self._format_as_text(filtered_df, table.get('name', ''))
                    processed_tables.append(processed_content)
                else:
                    processed_tables.append(table['raw_content'])
            except Exception as e:
                print(f"筛选表格行时出错: {e}")
                processed_tables.append(table['raw_content'])
                
        return "\n\n".join(processed_tables)
    
    def get_relevant_rows(self, df, table_name, question, max_rows=15):
        """
        使用VLLM来确定表格中与问题相关的行索引
        
        参数:
            df (pd.DataFrame): 表格数据
            table_name (str): 表格名称
            question (str): 问题文本
            max_rows (int): 返回的最大相关行索引数量
            
        返回:
            List[str]: 相关行索引列表，数量不超过max_rows
        """
        if df.empty:
            return []

        # 构建提示
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

        # 使用VLLM生成回答
        from vllm import SamplingParams
        
        # 准备输入提示
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 将消息转换为提示格式
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 如果tokenizer没有apply_chat_template方法，使用简单格式
            formatted_prompt = f"User: {prompt}\n\nAssistant:"
        
        # 使用VLLM生成回答
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.5,
            max_tokens=150,
            repetition_penalty=1.2,
            n=1
        )
        
        outputs = self.llm.generate(formatted_prompt, sampling_params)
        
        # 获取生成的文本
        response = outputs[0].outputs[0].text.strip()
        
        # 解析回答
        if "No relevant rows" in response:
            return ["No relevant rows"]
        
        # 提取逗号分隔的索引
        indices = []
        for item in response.split(','):
            item = item.strip()
            if item.isdigit():
                indices.append(item)
        
        # 限制返回的行数
        return indices[:max_rows]
    
    def _parse_tables(self, table_content):
        """解析表格内容为DataFrame"""
        # 简单实现，实际应用中可能需要更复杂的解析逻辑
        from .table_parser import parse_markdown_table
        import pandas as pd
        from io import StringIO
        
        tables = []
        
        # 尝试解析Markdown表格
        markdown_tables = parse_markdown_table(table_content)
        if markdown_tables:
            for i, table in enumerate(markdown_tables):
                tables.append({
                    'name': f"Table_{i+1}",
                    'df': table,
                    'raw_content': table_content,
                    'markdown': True
                })
            return tables
        
        # 尝试解析CSV格式
        try:
            df = pd.read_csv(StringIO(table_content))
            tables.append({
                'name': "Table_1",
                'df': df,
                'raw_content': table_content
            })
            return tables
        except:
            pass
        
        # 返回原始内容
        return [{
            'name': "Unknown",
            'df': None,
            'raw_content': table_content
        }]
    
    def _format_as_markdown(self, df, name):
        """将DataFrame格式化为Markdown表格"""
        result = f"## {name}\n\n"
        result += df.to_markdown(index=False)
        return result
    
    def _format_as_text(self, df, name):
        """将DataFrame格式化为文本表格"""
        result = f"## {name}\n\n"
        result += df.to_string(index=False)
        return result
    
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