import re
import torch
import pandas as pd
from io import StringIO
from .table_parser import parse_markdown_table

class SingleTableProcessor:
    """
    简单表格处理器
    仅对表格内容进行分词，不做任何预处理
    """
    
    def __init__(self, tokenizer, device="cuda:0", table_token_budget=2048):
        """
        初始化表格处理器
        
        参数:
            tokenizer: 分词器
            device: 设备名称
            table_token_budget: 表格token预算
        """
        self.tokenizer = tokenizer
        self.device = device
        self.table_token_budget = table_token_budget
    
    def process_table_content(self, db_str, question=None, use_llm_for_relevance=False,markdown=True):
        """
        简单处理表格内容：直接分词，不做任何预处理
        
        参数:
            db_str (str): 数据库表格字符串表示
            question (str, optional): 问题文本，此处不使用
            use_llm_for_relevance (bool, optional): 此处不使用
            
        返回:
            torch.Tensor: 表格内容的token ID
        """
        max_tokens = self.table_token_budget
        
        # 直接对原始表格字符串进行分词
        table_token_ids = self.tokenizer(
            db_str,
            return_tensors='pt',
            max_length=max_tokens,
            truncation=True,
            add_special_tokens=False
        ).input_ids
        
        if table_token_ids.shape[1] >= max_tokens:
            print(f"警告: 表格内容被截断至{max_tokens}个tokens。")
        
        return table_token_ids.to(self.device)