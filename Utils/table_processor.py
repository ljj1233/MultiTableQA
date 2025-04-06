import re
import torch
import pandas as pd
from io import StringIO
from .table_parser import parse_markdown_table

class TableProcessor:
    """
    表格处理器
    用于解析、过滤和线性化表格数据
    """
    
    def __init__(self, tokenizer, relevance_extractor, device="cuda:0", table_token_budget=2048):
        """
        初始化表格处理器
        
        参数:
            tokenizer: 分词器
            relevance_extractor: 表格相关行提取器
            device: 设备名称
            table_token_budget: 表格token预算
        """
        self.tokenizer = tokenizer
        self.relevance_extractor = relevance_extractor
        self.device = device
        self.table_token_budget = table_token_budget
    
    def process_table_content(self, db_str, question, use_llm_for_relevance=False):
        """
        解析多表格Markdown/CSV，提取相关模式和行（带列过滤），
        线性化，并在预算内标记化。

        参数:
            db_str (str): 数据库表格字符串表示。
            question (str): 问题文本。
            use_llm_for_relevance (bool): 启用基于LLM的行过滤的标志。

        返回:
            torch.Tensor or None: 处理后的表格token ID或失败时为None。
        """
        max_tokens = self.table_token_budget
        parsed_tables = []
        # 按'## table_name'标题拆分，保留标题
        table_sections = re.split(r'(^##\s+\w+\s*?$)', db_str, flags=re.MULTILINE)
        current_table_name = "default_table"
        content_buffer = []
        if len(table_sections) <= 1:
             content_buffer = db_str.strip().split('\n')
             table_sections = []
        else:
            for section in table_sections:
                section = section.strip()
                if not section: continue
                if section.startswith("##"):
                    if content_buffer:
                        # 使用迁移后的函数
                        df = parse_markdown_table(content_buffer)
                        if df is not None:
                             parsed_tables.append({"name": current_table_name, "df": df})
                        else: # 如果Markdown失败，尝试CSV
                            try:
                                delimiter = ',' if any(',' in line for line in content_buffer[:5]) else ('|' if any('|' in line for line in content_buffer[:5]) else '\t')
                                csv_like_string = "\n".join(line for line in content_buffer if line.strip()) # 跳过空行
                                df_csv = pd.read_csv(StringIO(csv_like_string), sep=delimiter, skipinitialspace=True, quotechar='"', on_bad_lines='skip')
                                if not df_csv.empty:
                                     df_csv = df_csv.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                                     parsed_tables.append({"name": current_table_name, "df": df_csv})
                                     print(f"信息: 将'{current_table_name}'下的块解析为CSV（分隔符'{delimiter}'）。")
                                else:
                                     print(f"警告: 无法将表'{current_table_name}'下的内容解析为Markdown或CSV。")
                            except Exception as e_csv:
                                print(f"警告: 解析表'{current_table_name}'下的内容为Markdown或CSV失败: {e_csv}")
                        content_buffer = []
                    current_table_name = section.lstrip('#').strip()
                else:
                    content_buffer.extend(section.split('\n'))
        # 处理最后一个块
        if content_buffer:
             df = parse_markdown_table(content_buffer)
             if df is not None:
                 parsed_tables.append({"name": current_table_name, "df": df})
             else: # 尝试CSV
                 try:
                     delimiter = ',' if any(',' in line for line in content_buffer[:5]) else ('|' if any('|' in line for line in content_buffer[:5]) else '\t')
                     csv_like_string = "\n".join(line for line in content_buffer if line.strip())
                     df_csv = pd.read_csv(StringIO(csv_like_string), sep=delimiter, skipinitialspace=True, quotechar='"', on_bad_lines='skip')
                     if not df_csv.empty:
                          df_csv = df_csv.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                          parsed_tables.append({"name": current_table_name, "df": df_csv})
                          print(f"信息: 将最后一个块'{current_table_name}'解析为CSV（分隔符'{delimiter}'）。")
                     else:
                          print(f"警告: 无法将'{current_table_name}'下的最后一个块解析为Markdown或CSV。")
                 except Exception as e_csv:
                     print(f"警告: 解析'{current_table_name}'下的最后一个块为Markdown或CSV失败: {e_csv}")

        # 如果没有解析到任何内容的回退
        if not parsed_tables:
            print("警告: 无法从db_str解析任何结构化表格。使用原始字符串（截断）。")
            encoded = self.tokenizer.encode(db_str, max_length=max_tokens, truncation=True)
            # 仅在需要时解码，直接返回张量
            return torch.tensor([encoded], device=self.device) # 直接返回张量


        # --- 4. 线性化与列和行过滤 ---
        compact_table_parts = []
        question_keywords = set(question.lower().split())
        rows_to_keep_per_table_fallback = 5 # 减少回退行计数

        for table_info in parsed_tables:
            table_name = table_info['name']
            df = table_info['df']

            if df.empty:
                continue

            # --- 步骤4a: 过滤列 ---
            selected_columns = self.relevance_extractor.get_relevant_columns(df, question_keywords)
            if not selected_columns:
                 print(f"警告: 未为表'{table_name}'识别到相关列。跳过表格。")
                 continue # 如果没有相关列，则跳过表格

            compact_table_parts.append(f"表格: {table_name}")
            # 使用选定的列作为模式
            schema_str = "模式: " + " | ".join(map(str,selected_columns))
            compact_table_parts.append(schema_str)

            # --- 步骤4b: 过滤行（使用LLM或关键词） ---
            relevant_indices = []
            if use_llm_for_relevance:
                # LLM过滤（使用优化的get_relevant_rows）
                # 根据df大小决定是否需要预过滤，然后调用LLM
                # 简化：现在直接调用LLM，假设_get_relevant_rows处理样本
                try:
                    relevant_indices = self.relevance_extractor.get_relevant_rows(df, table_name, question)
                    if relevant_indices:
                       print(f"信息: 使用LLM在表'{table_name}'中找到了{len(relevant_indices)}个相关行。")
                except Exception as e:
                    print(f"警告: LLM相关性筛选失败: {e}，将回退到关键词匹配。")
                    # 回退在下面处理

            # 关键词过滤（如果未使用LLM、LLM失败或返回空）
            if not relevant_indices:
                keyword_indices = []
                for index, row in df.iterrows():
                    try:
                        # 仅在选定的列中搜索关键词以提高效率
                        row_text_filtered_cols = ' '.join(map(str, row[selected_columns].fillna('').values)).lower()
                        if any(keyword in row_text_filtered_cols for keyword in question_keywords):
                            keyword_indices.append(index)
                    except Exception:
                        continue
                relevant_indices = keyword_indices # 使用关键词结果
                if relevant_indices and not use_llm_for_relevance: # 仅当关键词是主要方法时打印
                     print(f"信息: 使用关键词匹配在表'{table_name}'（相关列）中找到了{len(relevant_indices)}个相关行。")


            selected_indices = relevant_indices
            if not relevant_indices:
                # 回退：如果未找到相关行，保留前N行
                selected_indices = df.head(rows_to_keep_per_table_fallback).index.tolist()
                if selected_indices:
                    print(f"信息: 在表'{table_name}'中未找到相关行。使用前{len(selected_indices)}行。")

            # --- 步骤4c: 线性化选定的行和列 ---
            rows_added = 0
            # 限制实际线性化的行数，即使有很多相关行也避免过长
            MAX_ROWS_TO_LINEARIZE = 30 # 示例限制
            for index in selected_indices[:MAX_ROWS_TO_LINEARIZE]:
                 try:
                     # 仅选择特定行的相关列
                     row_data = df.loc[index, selected_columns]
                     row_values_str = [str(v) if pd.notna(v) else '' for v in row_data.values]
                     row_str = f"行 {index}: {' | '.join(row_values_str)}"
                     compact_table_parts.append(row_str)
                     rows_added += 1
                 except KeyError:
                     print(f"警告: 在表'{table_name}'的行线性化过程中未找到索引{index}。")
                 except Exception as e:
                     print(f"警告: 线性化表'{table_name}'的行{index}时出错: {e}")

            if rows_added == 0 and selected_indices:
                 print(f"警告: 无法线性化表'{table_name}'的任何选定行。")

            compact_table_parts.append("---") # 分隔符

        # --- 5. 标记化（保持不变） ---
        compact_table_str = "\n".join(compact_table_parts).strip().rstrip('---').strip()
        if not compact_table_str:
             print("警告: 生成的紧凑表格字符串为空。返回None。")
             return None
        
        table_token_ids = self.tokenizer(
            compact_table_str,
            return_tensors='pt',
            max_length=max_tokens,
            truncation=True,
            add_special_tokens=False
        ).input_ids
        if table_token_ids.shape[1] >= max_tokens:
            print(f"警告: 最终处理的表格内容被截断至{max_tokens}个tokens。")
        
        return table_token_ids.to(self.device)