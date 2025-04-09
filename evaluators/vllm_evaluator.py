import os
import torch
import re
import sqlite3
import time
from typing import List, Dict, Any, Optional, Union, Tuple

# VLLM相关导入
from vllm import LLM, SamplingParams

# 原始TableQAEvaluator相关导入
from Utils.dataLoader import TaskCore, DB,extractAnswer
from Utils.table_relevance import TableRelevanceExtractor
from Utils.table_parser import parse_markdown_table
from Utils.table_processor import TableProcessor
from Utils.table_processor_single import SingleTableProcessor
import sys
sys.path.append("../")
from symbolic import dataDict
from tqdm import tqdm

class VLLMTableQAEvaluator:
    """
    使用VLLM框架的表格问答评估器
    """
    
    def __init__(self, model_path, device="cuda:0", tensor_parallel_size=1, 
                 use_llm_for_relevance=False, max_model_len=8192):
        """
        初始化VLLM表格问答评估器
        
        参数:
        - model_path: 模型路径
        - device: 设备，例如"cuda:0"
        - tensor_parallel_size: 张量并行大小，用于多GPU推理
        - use_llm_for_relevance: 是否使用LLM进行表格相关性筛选
        - max_model_len: 模型最大长度
        """
        self.device = device
        self.model_path = model_path
        self.use_llm_for_relevance = use_llm_for_relevance
        self.table_token_budget = 5000
        self.markdown = True
        self.max_model_len = max_model_len
        
        # 初始化VLLM模型
        print(f"正在加载VLLM模型 {model_path}...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.85,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="half"  # 使用半精度浮点数以提高性能
        )
        print(f"VLLM模型 {model_path} 已加载完成")
        
        # 获取模型的tokenizer
        self.tokenizer = self.llm.get_tokenizer()
        
        # 初始化采样参数
        self.default_sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.5,
            max_tokens=2048,
            repetition_penalty=1.2,
            n=1
        )
        
        self.relevance_extractor = self._create_simple_relevance_extractor()
        
        self.single_table_processor = self._create_simple_table_processor()
        
        # 多表处理器
        self.table_processor = self._create_simple_multi_table_processor()
    
    def _create_simple_relevance_extractor(self):
        """创建一个基于VLLM的相关性提取器"""
        return TableRelevanceExtractor(self.llm, self.tokenizer, self.device)
    
    def _create_simple_table_processor(self):
        """创建一个基于VLLM的表格处理器"""
        return SingleTableProcessor(self.tokenizer, self.generate)
    
    def _create_simple_multi_table_processor(self):
        """创建一个基于VLLM的多表格处理器"""
        return TableProcessor(
            self.tokenizer, 
            self._create_simple_relevance_extractor(), 
            self,  # 传入自身作为LLM模型
            device=self.device, 
            table_token_budget=self.table_token_budget
        )
    
    def generate(self, prompt):
        """
        使用VLLM生成文本
        
        参数:
        - prompt: 提示文本
        
        返回:
        - 生成的文本
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that analyzes tables "
            },
            {
                "role": "user",
                "content": prompt
            }
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
            formatted_prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
        
        # 使用VLLM生成回答
        sampling_params = SamplingParams(
            temperature=0.1,   
            top_p=0.9,
            max_tokens=3000,
            n=1
        )
        
        outputs = self.llm.generate(formatted_prompt, sampling_params)
        
        # 获取生成的文本
        response = outputs[0].outputs[0].text
        
        return response.strip()
    
    def _load_prompt_templates(self, prompt_type="default"):
        """
        加载提示模板
        
        参数:
        - prompt_type: 提示类型，可选值为 "default", "cot", "retrace_table"
        
        返回:
        - 提示模板
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
    
    def answer_question(self, db_str, question, choices_str, meta_info=None, prompt_type="default", markdown=True):
        """
        使用VLLM回答问题
        
        参数:
        - db_str: 数据库字符串（表格内容）
        - question: 问题
        - choices_str: 选项字符串
        - meta_info: 元信息
        - prompt_type: 提示类型
        - markdown: 是否使用markdown格式
        
        返回:
        - 回答
        """
        # 使用表格处理器处理表格内容
        processed_db_str = db_str  # 默认使用原始db_str
        
        # 加载提示模板
        prompt_template = self._load_prompt_templates(prompt_type)
        full_prompt = prompt_template.format(db_str=processed_db_str, question=question)
        
        # 添加选项
        if choices_str and "{choices_str}" not in prompt_template:
            full_prompt += f"\n\n{choices_str}"
        elif choices_str:
            full_prompt = full_prompt.replace("{choices_str}", choices_str)
        
        # 准备输入提示
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that analyzes tables and answers questions accurately. Always provide your answer in the format 'Answer: X' where X is one of the given choices."
            },
            {
                "role": "user",
                "content": full_prompt.strip()
            }
        ]
        
        # 将消息转换为提示格式
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 如果tokenizer没有apply_chat_template方法，使用简单格式
            prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
        
        # 使用VLLM生成回答
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.5,
            max_tokens=5000,
            repetition_penalty=1.2,
            n=1
        )
        
        outputs = self.llm.generate(prompt, sampling_params)
        
        # 获取生成的文本
        response = outputs[0].outputs[0].text
        
        return response
    
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
        - prompt_type: 提示类型
        """
        # 初始化TaskCore
        task_core = TaskCore(db_root, task_path, result_path)
        self.markdown = markdown
        
        # 获取模型名称，根据提示类型添加后缀
        model_name = f"VLLM_TableLlama_{prompt_type}"
        
        # 创建一个包装函数，将prompt_type传递给answer_question
        def wrapped_answer_func(db_str, question, choices_str, meta_info=None):
            return self.answer_question(db_str, question, choices_str, meta_info, prompt_type=prompt_type, markdown=markdown)
        
        # 初始化评估指标和结果存储
        all_results = {}
        total_correct = 0
        total_questions = 0
        
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
                
                # 每个数据库和规模评估后，立即统计结果
                with sqlite3.connect(result_path) as conn:
                    cursor = conn.cursor()
                    # 获取当前数据库和规模的统计信息
                    cursor.execute(f"""
                        SELECT COUNT(*) as total,
                               SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                        FROM {dbn} 
                        WHERE scale = ? AND model = ?
                    """, (current_scale, model_name))
                    stats = cursor.fetchone()
                    if stats:
                        db_total = stats[0] if stats[0] else 0
                        db_correct = stats[1] if stats[1] else 0
                        
                        # 累计总数
                        total_questions += db_total
                        total_correct += db_correct
                        
                        # 存储每个数据库和规模的结果
                        key = f"{dbn}_{current_scale}"
                        all_results[key] = {
                            "database": dbn,
                            "scale": current_scale,
                            "total": db_total,
                            "correct": db_correct,
                            "accuracy": (db_correct / db_total * 100) if db_total > 0 else 0
                        }
        
        # 计算整体准确率
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        # 按规模统计准确率
        scale_accuracy = {}
        for key, result in all_results.items():
            scale = result["scale"]
            if scale not in scale_accuracy:
                scale_accuracy[scale] = {"total": 0, "correct": 0}
            
            scale_accuracy[scale]["total"] += result["total"]
            scale_accuracy[scale]["correct"] += result["correct"]
        
        # 计算每个规模的准确率
        for scale, counts in scale_accuracy.items():
            scale_accuracy[scale] = (counts["correct"] / counts["total"] * 100) if counts["total"] > 0 else 0
        
        print("\n=== 评估指标 ===")
        print(f"总问题数: {total_questions}")
        print(f"正确回答数: {total_correct}")
        print(f"整体准确率: {overall_accuracy:.2f}%")
        print("\n各规模准确率:")
        for scale, acc in scale_accuracy.items():
            print(f"{scale} 规模: {acc:.2f}%")
        print(f"\n评估完成，结果已保存到 {result_path}")
        
        # 返回所有统计指标
        return {
            "total_count": total_questions,
            "correct_count": total_correct,
            "overall_accuracy": overall_accuracy,
            "scale_accuracy": scale_accuracy
        }

    def batch_answer_questions(self, batch_data, prompt_type="default", markdown=True):
        """
        批量处理表格问答
        
        参数:
        - batch_data: 批量数据列表，每项包含 (db_str, question, choices_str, meta_info)
        - prompt_type: 提示类型
        - markdown: 是否使用markdown格式
        
        返回:
        - 批量回答结果列表
        """
        # 准备批量提示
        prompts = []
        for db_str, question, choices_str, meta_info in batch_data:
            # 加载提示模板
            prompt_template = self._load_prompt_templates(prompt_type)
            full_prompt = prompt_template.format(db_str=db_str, question=question)
            
            # 添加选项
            if choices_str and "{choices_str}" not in prompt_template:
                full_prompt += f"\n\n{choices_str}"
            elif choices_str:
                full_prompt = full_prompt.replace("{choices_str}", choices_str)
            
            # 准备输入提示
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that analyzes tables and answers questions accurately. Always provide your answer in the format 'Answer: X' where X is one of the given choices."
                },
                {
                    "role": "user",
                    "content": full_prompt.strip()
                }
            ]
            
            # 将消息转换为提示格式
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 如果tokenizer没有apply_chat_template方法，使用简单格式
                prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
            
            prompts.append(prompt)
        
        # 使用VLLM批量生成回答
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.5,
            max_tokens=5000,  # 对于大表格可能需要增加
            repetition_penalty=1.2,
            n=1
        )
        
        # 批量生成
        outputs = self.llm.generate(prompts, sampling_params)
        
        # 处理结果
        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)
        
        return responses

    def run_evaluation_with_batch(self, db_root, task_path, result_path, 
                        dataset_name, scale, markdown=True, 
                        db_limit=5, sample_limit=5, question_limit=5, 
                        time_sleep=0, prompt_type="default", batch_size=4):
        """
        使用批处理运行评估
        
        参数:
        - db_root: 数据库根目录
        - task_path: 任务文件路径
        - result_path: 结果保存路径
        - dataset_name: 数据集名称
        - scale: 数据规模
        - markdown: 是否使用markdown格式
        - db_limit, sample_limit, question_limit: 评估范围限制
        - time_sleep: 每次评估间隔时间
        - prompt_type: 提示类型
        - batch_size: 批处理大小
        """
        # 初始化TaskCore
        task_core = TaskCore(db_root, task_path, result_path)
        self.markdown = markdown
        
        # 获取模型名称
        model_name = f"VLLM_TableLlama_{prompt_type}"
        
        # 初始化评估指标和结果存储
        all_results = {}
        total_correct = 0
        total_questions = 0
        
        database_list = list(dataDict.keys())
        for dbn in tqdm(database_list, desc="database_list"):
            # 根据不同规模设置等待时间
            current_time_sleep = time_sleep
            if isinstance(scale, list):
                scale_list = scale
            else:
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
                
                # 使用批处理包装函数处理当前数据库和规模
                batch_wrapper(
                    task_core, dbn, model_name, current_scale, markdown,
                    db_limit, sample_limit, question_limit, batch_size, current_time_sleep
                )
                
                # 统计结果
                with sqlite3.connect(result_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"""
                        SELECT COUNT(*) as total,
                            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                        FROM {dbn} 
                        WHERE scale = ? AND model = ?
                    """, (current_scale, model_name))
                    stats = cursor.fetchone()
                    if stats:
                        db_total = stats[0] if stats[0] else 0
                        db_correct = stats[1] if stats[1] else 0
                        
                        # 累计总数
                        total_questions += db_total
                        total_correct += db_correct
                        
                        # 存储每个数据库和规模的结果
                        key = f"{dbn}_{current_scale}"
                        all_results[key] = {
                            "database": dbn,
                            "scale": current_scale,
                            "total": db_total,
                            "correct": db_correct,
                            "accuracy": (db_correct / db_total * 100) if db_total > 0 else 0
                        }
        
        # 计算整体准确率
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        # 按规模统计准确率
        scale_accuracy = {}
        for key, result in all_results.items():
            scale = result["scale"]
            if scale not in scale_accuracy:
                scale_accuracy[scale] = {"total": 0, "correct": 0}
            
            scale_accuracy[scale]["total"] += result["total"]
            scale_accuracy[scale]["correct"] += result["correct"]
        
        # 计算每个规模的准确率
        for scale, counts in scale_accuracy.items():
            scale_accuracy[scale] = (counts["correct"] / counts["total"] * 100) if counts["total"] > 0 else 0
        
        print("\n=== 评估指标 ===")
        print(f"总问题数: {total_questions}")
        print(f"正确回答数: {total_correct}")
        print(f"整体准确率: {overall_accuracy:.2f}%")
        print("\n各规模准确率:")
        for scale, acc in scale_accuracy.items():
            print(f"{scale} 规模: {acc:.2f}%")
        print(f"\n评估完成，结果已保存到 {result_path}")
        
        # 返回所有统计指标
        return {
            "total_count": total_questions,
            "correct_count": total_correct,
            "overall_accuracy": overall_accuracy,
            "scale_accuracy": scale_accuracy
        }

    def _process_batch(self, batch_data, batch_indices, dbn, model_name, scale, markdown, task_core):
        """处理一个批次的数据"""
        try:
            # 批量生成回答
            responses = self.batch_answer_questions(batch_data, prompt_type="default", markdown=markdown)
            
            # 处理每个回答
            for i, response in enumerate(responses):
                dbIdx, sampleIdx, questionIdx, gt = batch_indices[i]
                pred = extractAnswer(response)
                
                # 保存结果
                task_core.resultCur.execute(
                    TaskCore.inserttemplate.format(table_name=dbn),
                    (
                        model_name,
                        scale,
                        markdown,
                        dbIdx,
                        sampleIdx,
                        questionIdx,
                        gt,
                        pred,
                        gt == pred,
                        "",  # error
                        response,
                    ),
                )
            
            # 提交事务
            task_core.resultConn.commit()
            
        except Exception as e:
            print(f"批处理出错: {e}")
            # 单个处理回退
            for i, (db_str, question, choices_str, meta_info) in enumerate(batch_data):
                dbIdx, sampleIdx, questionIdx, gt = batch_indices[i]
                
                pred = ""
                error = ""
                res = ""
                try:
                    res = self.answer_question(db_str, question, choices_str, meta_info)
                    pred = extractAnswer(res)
                except Exception as e:
                    error = str(e)
                
                task_core.resultCur.execute(
                    TaskCore.inserttemplate.format(table_name=dbn),
                    (
                        model_name,
                        scale,
                        markdown,
                        dbIdx,
                        sampleIdx,
                        questionIdx,
                        gt,
                        pred,
                        gt == pred,
                        error,
                        res,
                    ),
                )
                task_core.resultConn.commit()