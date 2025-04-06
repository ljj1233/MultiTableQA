import os
import torch
import argparse
import re
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import time
import sys

sys.path.append(".")

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, GenerationConfig
from MTable import apply_table_llama, apply_table_function
from Utils.jsTool import JS
from eval.evaluator import evalAcc

class TableLlamaEvaluator:
    def __init__(self, model_path, device="cuda:0"):
        """
        初始化 TableLlama 模型
        
        Args:
            model_path: 模型路径
            device: 设备
        """
        self.device = device
        self.model_path = model_path

        # 应用表格函数增强
        apply_table_function()

        # 加载模型配置
        self.config = LlamaConfig.from_pretrained(model_path)
        self.config.rope_scaling = {
            "type": "linear",
            "factor": 2.0
        }
        
        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.float16
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token   
        
        # 初始化生成配置
        self.generation_config = GenerationConfig(
            max_length=1024,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 应用表格注入增强
        apply_table_llama(
            self.model,
            starting_layer=7,
            ending_layer=712,
            entropy_threshold=0.8,
            retracing_ratio=0.05
        )
        print(f"模型 {model_path} 已加载完成")
        
        # 加载提示模板
        self.prompt_templates = self._load_prompt_templates()

    def _load_prompt_templates(self):
        """
        加载提示模板
        
        Returns:
            提示模板字典
        """
        templates = {}
        template_files = {
            "default": "./prompts/default_prompt.txt",
            "cot": "./prompts/cot_prompt.txt",
            "retrace_table": "./prompts/retrace_table_prompt.txt"
        }
        
        # 确保提示目录存在
        os.makedirs("./prompts", exist_ok=True)
        
        # 加载提示模板
        for prompt_type, file_path in template_files.items():
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    templates[prompt_type] = f.read().strip()
            else:
                print(f"警告: 提示模板文件 {file_path} 不存在")
        
        return templates
    def process_table_content(self, db_str):
        """
        处理表格内容，将其转换为模型可用的格式
        
        Args:
            db_str: 数据库表格的字符串表示
            
        Returns:
            处理后的表格特征（当前实现为token IDs）
        """
        # 当前实现：直接将表格文本转换为token IDs
        # 后续可以在这里实现更复杂的表格特征提取逻辑
        table_token_ids = self.tokenizer(db_str, return_tensors='pt').input_ids
        return table_token_ids.to(self.device)
        
    def answer_question(self, db_str, question, choices_str, meta_info=None, prompt_type="default"):
        """
        回答问题
        
        Args:
            db_str: 数据库表格的字符串表示
            question: 问题文本
            choices_str: 选项字符串
            meta_info: 元信息(可选)
            prompt_type: 提示类型，可选值为 "default", "cot", "retrace_table"
            
        Returns:
            模型的回答
        """
        # 获取提示模板
        template = self.prompt_templates.get(prompt_type, self.prompt_templates["default"])
        
        # 填充模板
        full_prompt = template.format(db_str=db_str, question=question)
        
        # 如果有选项，添加到提示中
        if choices_str:
            full_prompt += f"\n\nThis question has only one correct answer. Please break down the question, evaluate each option, and explain why it is correct or incorrect.\n\n{choices_str}\n\nConclude with your final choice on a new line formatted as `Answer: A/B/C/D`."
        
        # 准备输入
        messages = [{"role": "user", "content": full_prompt}]
        print(f'full_prompt: {full_prompt}')
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,  
            truncation=True
        ).to(self.device)

        # 生成回答
        use_table_token = prompt_type == "retrace_table"  # 只在 retrace_table 模式下传入表格内容
        table_token_ids = self.process_table_content(db_str) if use_table_token else None

        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            table_token=table_token_ids,   
            tokenizer=self.tokenizer if use_table_token else None
        )

        # 解码回答
        response = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 提取用户提示之后的部分作为回答
        if "assistant" in response.lower():
            response = response.split("assistant", 1)[1].strip()
        
        return response

def main():
    parser = argparse.ArgumentParser(description="开源表格问答模型评估工具")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--dataset", type=str, required=True, help="数据集类型 (qa, fv, ret, cpa, cta, em, bqa)")
    parser.add_argument("--scale", type=str, default="16k", help="数据规模 (8k, 16k, 32k, 64k, 128k)")
    parser.add_argument("--markdown", action="store_true", help="使用markdown格式")
    parser.add_argument("--prompt_type", type=str, default="default", 
                        choices=["default", "cot", "retrace_table"],
                        help="提示类型: default(原始提问), cot(思维链), retrace_table(表格增强)")
    parser.add_argument("--log_root", type=str, default=None, help="日志根目录")
    parser.add_argument("--result_path", type=str, default=None, help="结果JSON路径")
    
    args = parser.parse_args()
    
    # 初始化模型
    evaluator = TableLlamaEvaluator(args.model_path)
    
    # 如果不使用表格增强功能，则禁用它
    if args.prompt_type != "retrace_table":
        # 禁用表格增强功能的代码
        for layer in evaluator.model.model.layers:
            if hasattr(layer.mlp, 'apply_table_injection'):
                layer.mlp.apply_table_injection = False
    
    # 运行评估
    evalAcc(
        ds=args.dataset,
        scale=args.scale,
        markdown=args.markdown,
        model=args.model_path,
        evaluator=evaluator,
        prompt_type=args.prompt_type,
        logRoot=args.log_root,
        resultPath=args.result_path,
        is_api_model=False
    )


# Single question test example
def test_single_question():
    model_path = "chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    evaluator = TableLlamaEvaluator(model_path)

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
    # 如果直接运行此脚本，则执行单个问题测试
    if not any('--' in arg for arg in os.sys.argv[1:]):
        test_single_question()
    else:
        main()