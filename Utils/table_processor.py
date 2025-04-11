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
    
    def __init__(self, tokenizer, relevance_extractor, llm_model, device="cuda:0", table_token_budget=2048):
        self.tokenizer = tokenizer
        self.relevance_extractor = relevance_extractor
        self.llm_model = llm_model  # 这里应该是VLLMTableQAEvaluator实例
        self.device = device
        self.table_token_budget = table_token_budget

    def generate_table_summary_and_filter_columns(self, table_name, df, question, filter_columns=False):
        """同时生成表格摘要和筛选相关列"""
        # 构建提示
        prompt = f"Please analyze the following table.\nTable name: {table_name}\n"
        prompt += f"Columns: {', '.join(df.columns)}\n"
        prompt += f"Sample data:\n{df.head(2).to_string()}\n"
        
        if filter_columns and len(df.columns) > 5:
            prompt += f"\nBased on the question: '{question}'\n"
            prompt += "Task 1: Provide a brief summary of the table's content and purpose in 1-2 sentences.\n"
            prompt += "Task 2: List only the column names that are relevant to answering this question, separated by commas.\n"
            prompt += "Format your response as:\nSummary: [your summary]\nRelevant columns: [comma-separated column names]"
        else:
            prompt += "\nTask: Provide a brief summary of the table's content and purpose in 1-2 sentences.\n"
            prompt += "Format your response as:\nSummary: [your summary]"
        
        response = self.llm_model.generate(prompt)
        
        # 解析响应
        summary = ""
        relevant_columns = []
        
        # 提取摘要
        summary_match = re.search(r"Summary:\s*(.*?)(?:\nRelevant columns:|$)", response, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
        
        # 如果需要筛选列，提取相关列
        if filter_columns and len(df.columns) > 5:
            columns_match = re.search(r"Relevant columns:\s*(.*?)$", response, re.DOTALL)
            if columns_match:
                column_text = columns_match.group(1).strip()
                relevant_columns = [col.strip() for col in column_text.split(',')]
                # 确保所有列名都在原表中存在
                relevant_columns = [col for col in relevant_columns if col in df.columns]
                
                # 如果没有找到有效列或筛选后列数太少，使用基本列
                all_columns = relevant_columns.copy()  # 初始化all_columns
                if len(relevant_columns) < 2:
                    # 至少保留ID列和一些基本信息列
                    basic_columns = []
                    for col in df.columns:
                        if 'id' in col.lower() or 'name' in col.lower() or 'code' in col.lower():
                            basic_columns.append(col)
                    
                    # 如果基本列加上有效列不足5个，添加更多列直到达到5个或用完所有列
                    all_columns = list(set(relevant_columns + basic_columns))
                    if len(all_columns) < 5:
                        remaining = list(set(df.columns) - set(all_columns))
                        all_columns.extend(remaining[:5-len(all_columns)])
                
                relevant_columns = all_columns if all_columns else list(df.columns)
    
        return summary, relevant_columns

    def _simplify_table_content(self, table_content, question):
        max_chars = self.table_token_budget * 4  # 粗略估计每个token约4个字符
        if len(table_content) > max_chars:
            return table_content[:max_chars] + "\n...(表格内容已截断)"
        
        return table_content

    def align_entities(self, parsed_tables, question):
        """
        识别多个表格之间的实体关系
        
        参数:
            parsed_tables: 解析后的表格列表
            question: 问题文本
            
        返回:
            实体对齐信息字符串
        """
        if len(parsed_tables) <= 1:
            return ""
            
        # 构建提示
        prompt = "Please identify the relationships between the following tables:\n\n"
        
        for table in parsed_tables:
            prompt += f"Table: {table['name']}\n"
            prompt += f"Columns: {', '.join(table['df'].columns)}\n"
            prompt += f"Sample data:\n{table['df'].head(2).to_string()}\n\n"
            
        prompt += f"Question: {question}\n\n"
        prompt += "Task: Identify the key columns that can be used to join these tables. Format your response as:\n"
        prompt += "Table1.column1 = Table2.column2\n"
        prompt += "Table2.column3 = Table3.column4\n"
        prompt += "etc."
        
        # 使用VLLM生成回答
        response = self.llm_model.generate(prompt)
        
        # 简单处理响应
        lines = response.strip().split('\n')
        valid_lines = []
        
        for line in lines:
            if '=' in line and any(table['name'] in line for table in parsed_tables):
                valid_lines.append(line.strip())
                
        return '\n'.join(valid_lines) if valid_lines else "No clear relationships identified"

    def process_table_content(self, db_str, question, use_llm_for_relevance=False, markdown=True):
        """处理表格内容，返回token和处理后的文本"""
        if not db_str or not isinstance(db_str, str):
            print(f"警告: 输入的db_str为空或类型不正确: {type(db_str)}")
            return torch.tensor([]).to(self.device), ""
                
        parsed_tables = []
        
        # 将文本按行分割
        lines = db_str.split('\n')
        main_sections = []
        current_section = []
        current_title = None
        
        # 逐行处理
        for line in lines:
            line = line.strip()
            if line.startswith('#') and not line.startswith('##'):
                # 如果遇到新的主标题，保存之前的部分
                if current_title:
                    main_sections.append((current_title, '\n'.join(current_section)))
                # 开始新的部分
                current_title = line.lstrip('#').strip()
                current_section = []
            else:
                current_section.append(line)
        
        # 处理最后一个部分
        if current_title:
            main_sections.append((current_title, '\n'.join(current_section)))
        
        # 处理每个部分
        for title, content in main_sections:
            if title and content:
                print(f"检测到主标题: {title}")
                tables = self._process_section(content, title)
                if tables:
                    parsed_tables.extend(tables)
                    print(f"成功解析表格，标题: {title}")
                else:
                    print(f"警告: 未能解析表格内容，标题: {title}")
    
        # 如果没有解析到任何表格，尝试直接解析整个内容
        if not parsed_tables:
            print("未解析到任何表格，尝试作为单个表格处理")
            try:
                df = pd.read_csv(StringIO(db_str))
                parsed_tables.append({"name": "default_table", "df": df})
            except Exception as e:
                print(f"直接解析失败: {str(e)}")
                return torch.tensor([]).to(self.device), db_str

        if len(main_sections) <= 1:
            # 处理无主标题情况，添加调试信息
            print("未检测到主标题，尝试直接解析表格内容")
            tables = self._process_section(db_str)
            if tables:
                parsed_tables.extend(tables)
            else:
                print("警告: 直接解析表格失败")
        else:
            # 处理有主标题情况
            current_main_title = None
            for section in main_sections:
                section = section.strip()
                if not section:
                    continue
                
                if section.startswith("#") and not section.startswith("##"):
                    current_main_title = section.lstrip('#').strip()
                    print(f"检测到主标题: {current_main_title}")
                else:
                    tables = self._process_section(section, current_main_title)
                    if tables:
                        parsed_tables.extend(tables)
                        print(f"成功解析表格，标题: {current_main_title}")
                    else:
                        print(f"警告: 未能解析表格内容，标题: {current_main_title}")
        
        # 如果没有解析到任何表格，尝试直接解析整个内容
        if not parsed_tables:
            print("未解析到任何表格，尝试作为单个表格处理")
            try:
                df = pd.read_csv(StringIO(db_str))
                parsed_tables.append({"name": "default_table", "df": df})
            except Exception as e:
                print(f"直接解析失败: {str(e)}")
                # 返回空结果
                return torch.tensor([]).to(self.device), db_str

        # 为每个表格生成摘要并筛选列（如果需要）
        for table in parsed_tables:
            summary, relevant_columns = self.generate_table_summary_and_filter_columns(
                table['name'], 
                table['df'], 
                question, 
                filter_columns=use_llm_for_relevance
            )
            
            table['summary'] = summary
            
            # 如果需要筛选列且有返回相关列
            if use_llm_for_relevance and relevant_columns:
                table['filtered_columns'] = relevant_columns
                table['df'] = table['df'][relevant_columns]
        
        # 进行实体对齐
        if len(parsed_tables) > 1:
            alignment = self.align_entities(parsed_tables, question)
            for table in parsed_tables:
                table['alignment'] = alignment
        
        # 构建最终的表格表示
        processed_content = []
        for table in parsed_tables:
            # 添加表格摘要
            processed_content.append(f"# {table['name']}")
            processed_content.append(f"Summary: {table['summary']}")
            
            # 如果进行了列筛选，添加筛选信息
            if 'filtered_columns' in table and use_llm_for_relevance:
                processed_content.append(f"Selected columns: {', '.join(table['filtered_columns'])}")
            
            if 'alignment' in table:
                processed_content.append(f"Entity Alignment: {table['alignment']}")
            
            # 添加表格内容
            processed_content.append(table['df'].to_string())
            processed_content.append("\n")
        
        # 生成最终文本
        final_text = "\n".join(processed_content)
        
        # 转换为token
        tokens = self.tokenizer.encode(final_text, add_special_tokens=False)
        
        # 确保不超过token预算
        if len(tokens) > self.table_token_budget:
            tokens = tokens[:self.table_token_budget]
            # 如果截断了tokens，也应该截断文本
            decoded_text = self.tokenizer.decode(tokens)
        else:
            decoded_text = final_text
        
        return torch.tensor(tokens).to(self.device), decoded_text

    def _process_section(self, section_content, main_title=None):
        """处理单个章节的表格"""
        if not section_content:
            return []
            
        tables = []
        # 使用更精确的正则表达式来匹配子标题
        sub_sections = re.split(r'(?m)^(##\s*[\w_]+)', section_content)
        current_table_name = main_title if main_title else "default_table"
        content_buffer = []

        print(f"处理章节，主标题: {main_title}")  # 调试信息

        for sub_section in sub_sections:
            sub_section = sub_section.strip()
            if not sub_section:
                continue

            if sub_section.startswith("##"):
                # 处理之前的内容
                if content_buffer:
                    print(f"尝试解析表格内容: {current_table_name}")  # 调试信息
                    print(f"内容缓冲区: {content_buffer[:2]}...")  # 显示部分内容
                    table = self._parse_table_content(content_buffer, current_table_name)
                    if table:
                        tables.append(table)
                        print(f"成功解析表格: {current_table_name}")
                    content_buffer = []
                current_table_name = sub_section.lstrip('#').strip()
            else:
                # 只保留非空且不以#开头的行
                valid_lines = []
                for line in sub_section.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        valid_lines.append(line)
                if valid_lines:
                    content_buffer.extend(valid_lines)

        # 处理最后一个表格
        if content_buffer:
            print(f"处理最后一个表格: {current_table_name}")  # 调试信息
            table = self._parse_table_content(content_buffer, current_table_name)
            if table:
                tables.append(table)
                print(f"成功解析最后一个表格: {current_table_name}")

        return tables

    def _parse_table_content(self, content_buffer, table_name):
        """解析表格内容"""
        try:
            # 首先尝试作为CSV解析
            content = '\n'.join(content_buffer)
            df = pd.read_csv(StringIO(content), skipinitialspace=True)
            print(f"成功通过CSV解析表格 {table_name}")  # 调试信息
            return {"name": table_name, "df": df}
        except Exception as e:
            print(f"CSV解析失败: {str(e)}")
            try:
                # 如果CSV解析失败，尝试Markdown解析
                df = parse_markdown_table(content_buffer)
                if df is not None:
                    print(f"成功通过Markdown解析表格 {table_name}")  # 调试信息
                    return {"name": table_name, "df": df}
            except Exception as e:
                print(f"Markdown解析失败: {str(e)}")
        return None


if __name__ == "__main__":
    # 导入必要的库
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from .table_relevance import TableRelevanceExtractor
    
    # 测试数据
    test_table_content = """
    # Employee Data

    ## Department
    Department_ID,Department_Name,Location
    1,HR,New York
    2,IT,San Francisco
    3,Finance,Chicago
    4,Marketing,Boston

    ## Employee
    Employee_ID,Name,Department_ID,Salary
    101,John Smith,1,50000
    102,Jane Doe,2,60000
    103,Bob Johnson,1,55000
    104,Alice Brown,3,65000
    """

    test_question = "How many employees work in the HR department?"

    try:
        # 初始化组件
        model_path = "/hpc2hdd/home/fye374/models/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 创建模型适配器
        model_adapter = ModelAdapter(model, tokenizer)
        
        relevance_extractor = TableRelevanceExtractor(model, tokenizer)
        
        # 创建TableProcessor实例
        processor = TableProcessor(
            tokenizer=tokenizer,
            relevance_extractor=relevance_extractor,
            llm_model=model_adapter,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            table_token_budget=2048
        )
        
        # 测试表格处理
        print("=== 测试表格处理 ===")
        tokens, processed_text = processor.process_table_content(
            test_table_content,
            test_question,
            use_llm_for_relevance=True,
            markdown=True
        )
        
        print("\n处理后的文本:")
        print(processed_text)
        
        print("\nToken数量:", len(tokens))
        
        # 测试表格摘要和列筛选
        print("\n=== 测试表格摘要和列筛选 ===")
        df = pd.read_csv(StringIO("""
            Employee_ID,Name,Department_ID,Salary,Email,Phone,Address
            101,John Smith,1,50000,john@company.com,123-456-7890,123 Main St
            102,Jane Doe,2,60000,jane@company.com,123-456-7891,456 Oak Ave
        """))
        
        summary, relevant_columns = processor.generate_table_summary_and_filter_columns(
            "Employee",
            df,
            test_question,
            filter_columns=True
        )
        
        print("\n生成的摘要:", summary)
        print("相关列:", relevant_columns)
        
        # 测试实体对齐
        print("\n=== 测试实体对齐 ===")
        parsed_tables = [
            {"name": "Department", "df": pd.read_csv(StringIO("Department_ID,Department_Name\n1,HR\n2,IT"))},
            {"name": "Employee", "df": pd.read_csv(StringIO("Employee_ID,Name,Department_ID\n101,John,1\n102,Jane,2"))}
        ]
        
        alignment = processor.align_entities(parsed_tables, test_question)
        print("\n实体对齐结果:", alignment)

    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")