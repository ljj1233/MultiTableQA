import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, GenerationConfig
from MTable import apply_table_llama, apply_table_function
from Utils.dataLoader import TaskCore
from symbolic import dataDict

from tqdm import tqdm
class TableQAEvaluator:
    def __init__(self, model_path, device="cuda:0", multi_gpu=False, use_llm_for_relevance=False):
        # 初始化 TableLlama 模型
        self.device = device
        self.multi_gpu = multi_gpu
        self.use_llm_for_relevance = use_llm_for_relevance
        self.table_token_budget = 2048  # 增加token预算以处理更多表格数据
        
        apply_table_function()

        # 加载模型配置
        self.config = LlamaConfig.from_pretrained(model_path)
        self.config.rope_scaling = {
            "type": "linear",
            "factor": 2.0
        }
        
        # 加载模型和分词器
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU 进行并行计算")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.float16,
                device_map="auto"  # 自动分配到可用的GPU上
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=self.config,
                torch_dtype=torch.float16
            ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token   
        
        # 初始化生成配置
        self.generation_config = GenerationConfig(
            max_length=800,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        apply_table_llama(
                self.model,
                starting_layer=8,
                ending_layer=12,
                entropy_threshold=0.9,
                retracing_ratio=0.02
            )
        print(f"模型 {model_path} 已加载完成")
  

    def _load_prompt_templates(self,prompt_type="default"):
        """
        加载提示模板
        
        Returns:
            提示模板
        """
        prompt_file_path = os.path.join("./prompts", f"{prompt_type}_prompt.txt")
        
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            print(f"警告: 未找到提示文件 {prompt_file_path}，使用默认提示")
            # 如果找不到文件，使用默认提示
            prompt_template = "Please carefully analyze and answer the following question:\n\n{db_str}\n\n{question}\n\nThis question has only one correct answer. Please break down the question, evaluate each option, and explain why it is correct or incorrect. Conclude with your final choice on a new line formatted as `Answer: A/B/C/D`."
        
        return prompt_template


    def run_evaluation(self, db_root, task_path, result_path, 
                      dataset_name, scale, markdown=True, 
                      db_limit=5, sample_limit=5, question_limit=5, 
                      time_sleep=0, prompt_type="default"):
        """
        运行评估
        
        参数:
        - db_root: 数据库根目录
        - task_path: 任务文件路径
        - result_path: 结果保存路径
        - dataset_name: 数据集名称
        - scale: 数据规模
        - markdown: 是否使用markdown格式
        - db_limit, sample_limit, question_limit: 评估范围限制
        - time_sleep: 每次评估间隔时间
        - prompt_type: 提示类型，可选值为 "default", "cot", "retrace_table"
        """
        # 初始化TaskCore
        task_core = TaskCore(db_root, task_path, result_path)
        
        # 获取模型名称，根据提示类型添加后缀
        model_name = f"TableLlama_{prompt_type}"
        
        # 创建一个包装函数，将prompt_type传递给answer_question
        def wrapped_answer_func(db_str, question, choices_str, meta_info=None):
            return self.answer_question(db_str, question, choices_str, meta_info, prompt_type=prompt_type)
        database_list = list(dataDict.keys())
        for dbn in tqdm(database_list, desc="database_list"):
            # 根据不同规模设置等待时间
            current_time_sleep = time_sleep
            if isinstance(scale, list):
                # 如果scale是列表，使用传入的scale列表
                scale_list = scale
            else:
                # 如果scale是单个值，转换为列表
                scale_list = [scale]
            
            for current_scale in scale_list:
                # 根据不同规模设置等待时间
                if current_scale == '16k':
                    current_time_sleep = 20
                elif current_scale == '32k':
                    current_time_sleep = 30
                elif current_scale == '64k':
                    current_time_sleep = 40
                else:
                    current_time_sleep = 5
                # 运行评估
                task_core.testAll(
                    model=model_name,
                    dbn=dbn,
                    scale=current_scale,
                    markdown=markdown,
                    dbLimit=db_limit,
                    sampleLimit=sample_limit,
                    questionLimit=question_limit,
                    func=wrapped_answer_func,
                    timeSleep=current_time_sleep
                )
            
        print(f"评估完成，结果已保存到 {result_path}")


    def _get_relevant_rows_with_llm(self, df, table_name, question, max_rows=15):
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
        # 选项2: 受控采样（如果之前的设置效果良好，可以保留）
        # simple_gen_config = GenerationConfig(
        #     max_new_tokens=150,
        #     temperature=0.3, # 如果采样，温度略高
        #     top_p=0.9,       # 略微收紧top_p
        #     do_sample=True
        # )

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

            # print(f"调试: 表'{table_name}'的LLM相关性输出: {output_text}") # 调试打印

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


    def _get_relevant_columns(self, df, question_keywords, include_ids=True):
        """
        Identifies relevant columns based on keyword matching with column names.
        Optionally always includes likely ID/Name columns.

        Args:
            df (pd.DataFrame): The table data.
            question_keywords (set): Set of lowercase keywords from the question.
            include_ids (bool): Whether to always include potential ID/Name columns.

        Returns:
            List[str]: A list of relevant column names.
        """
        relevant_cols = set()
        potential_id_cols = set()

        if df.empty:
            return []

        for col in df.columns:
            col_lower = str(col).lower() # Ensure column name is string and lowercased

            # Basic check if column seems like an ID/Name (heuristic)
            if include_ids and ('id' in col_lower or 'key' in col_lower or 'name' in col_lower or 'identifier' in col_lower or 'code' in col_lower):
                 potential_id_cols.add(col)

            # Check if column name itself is relevant
            # Split column name in case it's multi-word (e.g., "department_name")
            col_name_parts = set(re.split(r'[_\s-]+', col_lower))
            if question_keywords.intersection(col_name_parts):
                 relevant_cols.add(col)

            # Optional: Check content (can be slow, use cautiously)
            # Consider adding a flag to enable/disable this
            # if df[col].dtype == 'object':
            #     try:
            #         # Check a sample for performance
            #         sample_size = min(len(df), 50)
            #         if any(df[col].head(sample_size).str.contains(keyword, case=False, na=False).any() for keyword in question_keywords):
            #              relevant_cols.add(col)
            #     except Exception:
            #         pass

        # Combine relevant columns and potential ID columns
        final_relevant_cols = relevant_cols.union(potential_id_cols if include_ids else set())


        # Fallback: If no columns identified, use potential IDs or just the first column
        if not final_relevant_cols:
            if potential_id_cols:
                 final_relevant_cols = potential_id_cols
            elif len(df.columns) > 0:
                 final_relevant_cols = {df.columns[0]} # Fallback to first column

        # Return in original order, ensuring at least one column if possible
        ordered_final_cols = [col for col in df.columns if col in final_relevant_cols]

        if not ordered_final_cols and len(df.columns) > 0:
             return [df.columns[0]] # Absolute fallback: first column
        elif not ordered_final_cols:
             return [] # No columns in df
        else:
             return ordered_final_cols


    def _parse_markdown_table(self, table_lines):
        """Attempts to parse a simple Markdown table."""
        header = []
        data = []
        separator_found = False
        header_parsed = False

        for i, line in enumerate(table_lines):
            line = line.strip()
            if not line:
                continue

            # Look for separator line (e.g., |---|---|)
            if re.match(r'^[|\s]*[-:|]+[|\s]*$', line):
                if header_parsed: # Separator must come after header
                    separator_found = True
                continue # Skip separator line from data

            # If it's not the separator, parse as header or data
            if '|' in line:
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                if not header_parsed and not separator_found:
                    header = cells
                    header_parsed = True
                elif header_parsed and separator_found:
                    # Ensure row has same number of cells as header (or handle mismatch)
                    if len(cells) == len(header):
                       data.append(cells)
                    else:
                       print(f"Warning: Row data mismatch in Markdown table. Header has {len(header)} cols, row has {len(cells)}. Skipping row: {line}")

        if header and data:
            try:
                return pd.DataFrame(data, columns=header)
            except Exception as e:
                print(f"Warning: Failed to create DataFrame from parsed Markdown: {e}")
                return None
        return None

    def process_table_content(self, db_str, question, use_llm_for_relevance=False): # Added use_llm_for_relevance flag
        """
        Parses multi-table Markdown/CSV, extracts relevant schema & rows (with column filtering),
        linearizes, and tokenizes within budget.

        Args:
            db_str (str): Database table(s) string representation.
            question (str): The question text.
            use_llm_for_relevance (bool): Flag to enable LLM-based row filtering.

        Returns:
            torch.Tensor or None: Processed table token IDs or None on failure.
        """
        max_tokens = self.table_token_budget
        parsed_tables = []
        # --- Parsing Logic (Sections 1, 2, 3 - remains mostly the same) ---
        # ... (Keep the robust parsing logic for ## headers, Markdown, and CSV fallback) ...
        # Split by '## table_name' headers, keeping the headers
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
                        df = self._parse_markdown_table(content_buffer)
                        if df is not None:
                             parsed_tables.append({"name": current_table_name, "df": df})
                        else: # Try CSV if Markdown failed
                            try:
                                delimiter = ',' if any(',' in line for line in content_buffer[:5]) else ('|' if any('|' in line for line in content_buffer[:5]) else '\t')
                                csv_like_string = "\n".join(line for line in content_buffer if line.strip()) # Skip empty lines
                                df_csv = pd.read_csv(StringIO(csv_like_string), sep=delimiter, skipinitialspace=True, quotechar='"', on_bad_lines='skip')
                                if not df_csv.empty:
                                     df_csv = df_csv.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                                     parsed_tables.append({"name": current_table_name, "df": df_csv})
                                     print(f"Info: Parsed block under '{current_table_name}' as CSV (delimiter '{delimiter}').")
                                else:
                                     print(f"Warning: Failed to parse content under table '{current_table_name}' as Markdown or CSV.")
                            except Exception as e_csv:
                                print(f"Warning: Failed parsing content under '{current_table_name}' as Markdown or CSV: {e_csv}")
                        content_buffer = []
                    current_table_name = section.lstrip('#').strip()
                else:
                    content_buffer.extend(section.split('\n'))
        # Process last block
        if content_buffer:
             df = self._parse_markdown_table(content_buffer)
             if df is not None:
                 parsed_tables.append({"name": current_table_name, "df": df})
             else: # Try CSV
                 try:
                     delimiter = ',' if any(',' in line for line in content_buffer[:5]) else ('|' if any('|' in line for line in content_buffer[:5]) else '\t')
                     csv_like_string = "\n".join(line for line in content_buffer if line.strip())
                     df_csv = pd.read_csv(StringIO(csv_like_string), sep=delimiter, skipinitialspace=True, quotechar='"', on_bad_lines='skip')
                     if not df_csv.empty:
                          df_csv = df_csv.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                          parsed_tables.append({"name": current_table_name, "df": df_csv})
                          print(f"Info: Parsed last block '{current_table_name}' as CSV (delimiter '{delimiter}').")
                     else:
                          print(f"Warning: Failed to parse last block under '{current_table_name}' as Markdown or CSV.")
                 except Exception as e_csv:
                     print(f"Warning: Failed parsing last block under '{current_table_name}' as Markdown or CSV: {e_csv}")

        # Fallback if nothing parsed
        if not parsed_tables:
            print("Warning: Could not parse any structured tables from db_str. Using raw string (truncated).")
            encoded = self.tokenizer.encode(db_str, max_length=max_tokens, truncation=True)
            # Decode only if needed, return tensor directly
            return torch.tensor([encoded], device=self.device) # Return tensor directly


        # --- 4. Linearization with Column and Row Filtering ---
        compact_table_parts = []
        question_keywords = set(question.lower().split())
        rows_to_keep_per_table_fallback = 5 # Reduced fallback row count

        for table_info in parsed_tables:
            table_name = table_info['name']
            df = table_info['df']

            if df.empty:
                continue

            # --- Step 4a: Filter Columns ---
            selected_columns = self._get_relevant_columns(df, question_keywords)
            if not selected_columns:
                 print(f"Warning: No relevant columns identified for table '{table_name}'. Skipping table.")
                 continue # Skip table if no columns are relevant

            compact_table_parts.append(f"表格: {table_name}")
            # Use selected columns for schema
            schema_str = "模式: " + " | ".join(map(str,selected_columns))
            compact_table_parts.append(schema_str)

            # --- Step 4b: Filter Rows (using LLM or Keywords) ---
            relevant_indices = []
            if use_llm_for_relevance:
                # LLM Filtering (use the refined _get_relevant_rows_with_llm)
                # Decide if pre-filtering is needed based on df size before calling LLM
                # Simplified: Call LLM directly for now, assuming _get handles samples
                try:
                    relevant_indices = self._get_relevant_rows_with_llm(df, table_name, question)
                    if relevant_indices:
                       print(f"信息: 使用LLM在表'{table_name}'中找到了{len(relevant_indices)}个相关行。")
                except Exception as e:
                    print(f"警告: LLM相关性筛选失败: {e}，将回退到关键词匹配。")
                    # Fallback handled below

            # Keyword Filtering (if LLM not used, failed, or returned empty)
            if not relevant_indices:
                keyword_indices = []
                for index, row in df.iterrows():
                    try:
                        # Search for keywords only within the selected columns for efficiency
                        row_text_filtered_cols = ' '.join(map(str, row[selected_columns].fillna('').values)).lower()
                        if any(keyword in row_text_filtered_cols for keyword in question_keywords):
                            keyword_indices.append(index)
                    except Exception:
                        continue
                relevant_indices = keyword_indices # Use keyword results
                if relevant_indices and not use_llm_for_relevance: # Print only if keywords were the primary method
                     print(f"信息: 使用关键词匹配在表'{table_name}'（相关列）中找到了{len(relevant_indices)}个相关行。")


            selected_indices = relevant_indices
            if not relevant_indices:
                # Fallback: Keep top N rows if no relevant rows found
                selected_indices = df.head(rows_to_keep_per_table_fallback).index.tolist()
                if selected_indices:
                    print(f"信息: 在表'{table_name}'中未找到相关行。使用前{len(selected_indices)}行。")

            # --- Step 4c: Linearize Selected Rows and Columns ---
            rows_added = 0
            # Limit the number of rows actually linearized to avoid excessive length even if many are relevant
            MAX_ROWS_TO_LINEARIZE = 30 # Example limit
            for index in selected_indices[:MAX_ROWS_TO_LINEARIZE]:
                 try:
                     # Select only the relevant columns for the specific row
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
                 print(f"Warning: Failed to linearize any selected rows for table '{table_name}'.")

            compact_table_parts.append("---") # Separator

        # --- 5. Tokenization (Remains the same) ---
        compact_table_str = "\n".join(compact_table_parts).strip().rstrip('---').strip()
        if not compact_table_str:
             print("Warning: Generated compact table string is empty. Returning None.")
             return None
        # print(f"--- Compact Table String (Cols+Rows Filtered) ---\n{compact_table_str}\n------------------------------------------") # Debug
        table_token_ids = self.tokenizer(
            compact_table_str,
            return_tensors='pt',
            max_length=max_tokens,
            truncation=True,
            add_special_tokens=False
        ).input_ids
        if table_token_ids.shape[1] >= max_tokens:
            print(f"Warning: Final processed table content was truncated to {max_tokens} tokens.")
        # print(f"--- Final Table Token IDs ---\n{table_token_ids}\nShape: {table_token_ids.shape}\n--------------------------------") # Debug
        return table_token_ids
    
    
    def answer_question(self, db_str, question, choices_str, meta_info=None, prompt_type="default"):
        """
        回答问题 (Modified to pass question to process_table_content)
        """
        # ... (prompt loading and formatting remains the same) ...
        prompt_template = self._load_prompt_templates(prompt_type)
        # Make sure db_str in the prompt is the *original* one if needed by the template
        # Or decide if the template should use the processed compact string (less likely)
        full_prompt = prompt_template.format(db_str=db_str, question=question) # Use original db_str in prompt

        if choices_str and "{choices_str}" not in prompt_template:
            full_prompt += f"\n\n{choices_str}"
        elif choices_str:
            full_prompt = full_prompt.replace("{choices_str}", choices_str)

        # Prepare input prompt tokens
        messages = [{"role": "user", "content": full_prompt}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            # Truncation here applies to the *entire* input (prompt + potential space for answer)
            # Consider max_length carefully based on model limits and expected answer length
            truncation=True,
            max_length=self.model.config.max_position_embeddings - self.generation_config.max_length # Reserve space for generation
        ).to(self.device)

        # --- Generate Table Token IDs for Injection ---
        table_token_ids = None
        use_table_token = prompt_type == "retrace_table"
        if use_table_token:
             # Pass the original db_str and the question to the processing function
             table_token_ids = self.process_table_content(db_str, question)

             # Ensure table_token_ids are on the correct device if not None
             if table_token_ids is not None:
                 table_token_ids = table_token_ids.to(self.device)


        # --- Generate Answer ---
        # Ensure table_token is handled correctly by your modified model.generate
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.eos_token_id, # Use EOS for padding during generation
            max_new_tokens=800, # Controlled by generation_config.max_length now? Check precedence. Set max_new_tokens explicitly if needed.
            temperature=0.85,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.0,
            # Pass the processed table tokens *only* when using retrace_table
            table_token=table_token_ids if use_table_token else None,
            # Pass tokenizer only if needed by injection mechanism during generation (unlikely needed here)
            tokenizer=self.tokenizer if use_table_token else None
        )

        # Decode the *generated part* only
        # inputs.input_ids.shape[1] gives the length of the prompt
        output_token_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(
            output_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )


        return response.strip()


def main():
    parser = argparse.ArgumentParser(description="表格问答评估工具")
    parser.add_argument("--model_path", type=str, default="chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct", 
                        help="模型路径")
    parser.add_argument("--db_root", type=str, required=True, help="数据库根目录")
    parser.add_argument("--task_path", type=str, required=True, help="任务文件路径")
    parser.add_argument("--result_path", type=str, required=True, help="结果保存路径")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--scale", type=str, nargs='+', default=["8k"], 
                        choices=["8k", "16k", "32k", "64k", "128k"],
                        help="数据规模，可指定多个值，如: 8k 16k 32k")
    parser.add_argument("--markdown", action="store_true", help="使用markdown格式")
    parser.add_argument("--db_limit", type=int, default=5, help="数据库数量限制")
    parser.add_argument("--sample_limit", type=int, default=5, help="每个数据库的样本数量限制")
    parser.add_argument("--question_limit", type=int, default=5, help="每个样本的问题数量限制")
    parser.add_argument("--time_sleep", type=float, default=0, help="每次评估间隔时间")
    parser.add_argument("--prompt_type", type=str, default="default", 
                        choices=["default", "cot", "retrace_table"],
                        help="提示类型: default(原始提问), cot(思维链), retrace_table(表格增强)")
    parser.add_argument("--device", type=str, default="cuda:0", help="指定使用的设备，例如 'cuda:0'")
    parser.add_argument("--multi_gpu", action="store_true", help="是否使用多GPU并行计算")
    parser.add_argument("--use_llm_relevance", action="store_true", 
                        help="使用LLM进行表格相关性筛选（可能会增加处理时间）")
    
    args = parser.parse_args()
    
    # 初始化评估器，传入设备和多GPU参数
    evaluator = TableQAEvaluator(
        args.model_path, 
        device=args.device, 
        multi_gpu=args.multi_gpu,
        use_llm_for_relevance=args.use_llm_relevance
    )
    
    # 如果不使用表格增强功能，则禁用它
    if args.prompt_type != "retrace_table":
        for layer in evaluator.model.model.layers:
            if hasattr(layer.mlp, 'apply_table_injection'):
                layer.mlp.apply_table_injection = False
    
    # 对每个scale进行评估
    for scale in args.scale:
        # 根据scale设置time_sleep
        time_sleep = 0
        if scale == "16k":
            time_sleep = 30
        elif scale == "32k":
            time_sleep = 60
            
        print(f"\n开始评估 scale={scale}")
        # 运行评估
        evaluator.run_evaluation(
            db_root=args.db_root,
            task_path=args.task_path,
            result_path=args.result_path.replace('.sqlite', f'_{scale}.sqlite'),  
            dataset_name=args.dataset,
            scale=scale,
            markdown=args.markdown,
            db_limit=args.db_limit,
            sample_limit=args.sample_limit,
            question_limit=args.question_limit,
            time_sleep=time_sleep or args.time_sleep,   
            prompt_type=args.prompt_type
        )


# Single question test example
def test_single_question():
    model_path = "chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # 检测是否有多个GPU可用
    multi_gpu = torch.cuda.device_count() > 1
    evaluator = TableQAEvaluator(model_path, multi_gpu=multi_gpu)

    # Table content for multi-table association
    table_content = """
    ## departments
    | department_id | department_name | location    | manager_id |
    |---------------|----------------|-------------|------------|
    | 101           | R & D Department | Beijing     | 1          |
    | 102           | Sales Department | Shanghai    | 3          |
    | 103           | Finance Department | Guangzhou  | 5          |

    ## projects
    | project_id | project_name | department_id | start_date  | end_date    | budget  |
    |------------|--------------|---------------|-------------|-------------|---------|
    | 201        | Product A Development | 101           | 2023-01-15  | 2023-06-30  | 500  |
    | 202        | Marketing Promotion | 102           | 2023-02-01  | 2023-04-30    | 300  |
    | 203        | Finance System Upgrade | 103           | 2023-03-10  | 2023-05-15 | 250  |
    | 204        | Product B Development | 101           | 2023-04-01  | 2023-09-30  | 600  |
    """

    # Question for multi-table association
    # question = "What is the total budget of the projects managed by the R & D Department? Choices:\n A. 1100\nB. 670\nC. 500 \nD. 1650"
    question = "Which project is managed by the Sales Department? Choices:\n A. Product A Development\nB. Marketing Promotion\nC. Finance System Upgrade \nD. Product B Development"


    '''
    正确答案
    是C
    '''

    # Test three different question - asking methods
    for prompt_type in ["default", "cot", "retrace_table"]:
        print(f"\n===== Question - asking method: {prompt_type} =====")
        response = evaluator.answer_question(table_content, question, "", prompt_type=prompt_type)
        print("Answer:", response)




if __name__ == "__main__":
    # If this script is run directly, execute the single question test
    if not any('--' in arg for arg in os.sys.argv[1:]):
        test_single_question()
    else:
        main()