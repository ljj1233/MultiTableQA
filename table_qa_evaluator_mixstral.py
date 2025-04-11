import os
import torch
import argparse
import re
import sqlite3  
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from MTable import apply_table_function_mistral, apply_table_llama
from Utils.dataLoader import TaskCore
from Utils.table_processor_single import SingleTableProcessor  

from symbolic import dataDict
import pandas as pd
from io import StringIO

from tqdm import tqdm

class TableQAEvaluator:
    def __init__(self, model_path, device="cuda:0", multi_gpu=False, use_llm_for_relevance=False):
        # 初始化设备
        self.device = device
        self.multi_gpu = multi_gpu
        self.use_llm_for_relevance = use_llm_for_relevance
        self.table_token_budget = 20000   
        
        # 应用表格增强功能
        apply_table_function_mistral()
        
        # 加载分词器
        print(f"正在加载Mistral模型分词器: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        # 设置填充标记和最大长度
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 设置模型的最大长度
        self.max_length = min(self.table_token_budget, 18000)  # 使用较小的值
        self.tokenizer.model_max_length = self.max_length

        # 加载模型
        print(f"正在加载Mistral模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        # 初始化表格处理器
        self.table_processor = SingleTableProcessor(self.tokenizer, self.device, self.table_token_budget)
        
        # 应用表格增强功能到模型
        apply_table_llama(
            self.model,
            starting_layer=15,
            ending_layer=17,
            entropy_threshold=0.9,
            retracing_ratio=0.05
        )
        print(f"模型 {model_path} 已加载完成")

    def _load_prompt_templates(self, prompt_type="default"):
        """加载提示模板"""
        prompt_file_path = os.path.join("./prompts", f"{prompt_type}_prompt.txt")
        
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            print(f"警告: 未找到提示文件 {prompt_file_path}，使用默认提示")
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
        self.markdown=markdown
        # 获取模型名称，根据提示类型添加后缀
        model_name = f"TableLlama_{prompt_type}"
        
        # 创建一个包装函数，将prompt_type传递给answer_question
        def wrapped_answer_func(db_str, question, choices_str, meta_info=None):
            return self.answer_question(db_str, question, choices_str, meta_info, prompt_type=prompt_type,markdown=markdown)
        
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
   
    
    def answer_question(self, db_str, question, choices_str, meta_info=None, prompt_type="default"):
        """回答问题"""
        prompt_template = self._load_prompt_templates(prompt_type)
        full_prompt = prompt_template.format(db_str=db_str, question=question)

        if choices_str and "{choices_str}" not in prompt_template:
            full_prompt += f"\n\n{choices_str}"
        elif choices_str:
            full_prompt = full_prompt.replace("{choices_str}", choices_str)

        # 准备对话消息
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that analyzes tables"},
            {"role": "user", "content": full_prompt}
        ]
        
        # 使用 tokenizer 的 chat template
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理输入
        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,  # 添加这行
            return_attention_mask=True
        ).to(self.device)
        
        # 处理表格内容
        table_token_ids = None
        use_table_token = prompt_type == "retrace_table"
        if use_table_token:
            table_token_ids = self.table_processor.process_table_content(db_str, question, self.use_llm_for_relevance)
            if table_token_ids is not None:
                table_token_ids = table_token_ids.to(self.device)

        # 生成配置
        gen_kwargs = {
            "max_new_tokens": 3000,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.5,
            "repetition_penalty": 1.2,
            "table_token": table_token_ids if use_table_token else None,
            "tokenizer": self.tokenizer if use_table_token else None
        }
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response

def main():
    parser = argparse.ArgumentParser(description="多表格问答评估")
    parser.add_argument("--model_path", type=str, default="GLM/", 
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
                        help="使用LLM进行表格相关性筛选(可能会增加处理时间)")
    
    args = parser.parse_args()
    
    # 初始化评估器
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
    
    # 存储所有scale的评估结果
    all_results = {'markdown': {}, 'csv': {}}
    
    # 对每个scale进行评估
    for scale in args.scale:
        # 根据scale设置time_sleep
        time_sleep = 0
        if scale == "16k":
            time_sleep = 30
        elif scale == "32k":
            time_sleep = 60
        
        # 分别评估 markdown 和非 markdown 格式
        for format_type in ['markdown', 'csv']:
            is_markdown = format_type == 'markdown'
            print(f"\n开始评估 scale={scale}, format={format_type}")
            
            # 修改结果文件路径以区分格式
            current_result_path = args.result_path.replace(
                '.sqlite', 
                f'_{scale}_{format_type}.sqlite'
            )
            
            # 运行评估并获取指标
            metrics = evaluator.run_evaluation(
                db_root=args.db_root,
                task_path=args.task_path,
                result_path=current_result_path,
                dataset_name=args.dataset,
                scale=scale,
                markdown=is_markdown,   
                db_limit=args.db_limit,
                sample_limit=args.sample_limit,
                question_limit=args.question_limit,
                time_sleep=time_sleep or args.time_sleep,
                prompt_type=args.prompt_type
            )
            
            all_results[format_type][scale] = metrics

    # 分别输出 markdown 和非 markdown 的评估结果
    for format_type in ['markdown', 'csv']:
        print(f"\n=== {format_type} 格式评估结果总结 ===")
        total_correct = 0
        total_questions = 0
        
        for scale, metrics in all_results[format_type].items():
            print(f"\nScale: {scale}")
            print(f"总问题数: {metrics['total_count']}")
            print(f"正确回答数: {metrics['correct_count']}")
            # 计算准确率
            accuracy = (metrics['correct_count'] / metrics['total_count'] * 100) if metrics['total_count'] > 0 else 0
            print(f"准确率: {accuracy:.2f}%")
            
            # 累计总数
            total_correct += metrics['correct_count']
            total_questions += metrics['total_count']
        
        # 计算整体准确率
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        print(f"\n=== {format_type} 格式整体统计 ===")
        print(f"总问题数: {total_questions}")
        print(f"总正确数: {total_correct}")
        print(f"整体准确率: {overall_accuracy:.2f}%")
        print(f"{format_type} 格式各规模准确率:")
        for scale, metrics in all_results[format_type].items():
            for sub_scale, acc in metrics['scale_accuracy'].items():
                print(f"  {sub_scale} 规模: {acc:.2f}%")

# 单个问题测试示例
def test_single_question():
    model_path = "/hpc2hdd/home/fye374/models/Mistral-7B-Instruct-v0.3"  # 使用官方模型路径
    evaluator = TableQAEvaluator(model_path)
    
    # 表格内容
    table_content = """
    # Employee Information Table
    
    ## Employees
    
    EmployeeID,Name,DepartmentID,Position,Salary,JoinDate
    1,Zhang San,101,Engineer,15000,2020-01-15
    2,Li Si,102,Designer,12000,2019-05-20
    3,Wang Wu,101,Senior Engineer,20000,2018-03-10
    4,Zhao Liu,103,Product Manager,18000,2021-02-01
    5,Qian Qi,102,UI Designer,13000,2020-07-15
    
    ## Departments
    
    DepartmentID,DepartmentName,Manager,Location
    101,R&D Department,Zhang Ming,Building A 3F
    102,Design Department,Liu Fang,Building A 2F
    103,Product Department,Chen Qiang,Building B 1F
    
    ## Projects
    
    ProjectID,ProjectName,ResponsibleDept,Budget,StartDate,EndDate
    P001,Mobile App Development,101,500000,2022-01-01,2022-06-30
    P002,Website Redesign,102,300000,2022-02-15,2022-05-15
    P003,New Product Planning,103,450000,2022-03-01,2022-08-31
    """

    # Example question
    question = "How many employees are in the R&D Department?"
    choices_str = "A. 1 person\nB. 2 persons\nC. 3 persons\nD. 4 persons"

    # 测试不同问题提问方式
    for prompt_type in ["default", "cot", "retrace_table"]:
        print(f"\n===== 问题提问方式: {prompt_type} =====")
        response = evaluator.answer_question(table_content, question, choices_str, prompt_type=prompt_type)
        print("回答:", response)

if __name__ == "__main__":
    # 如果直接运行此脚本，执行单个问题测试
    if not any('--' in arg for arg in os.sys.argv[1:]):
        test_single_question()
    else:
        main()
