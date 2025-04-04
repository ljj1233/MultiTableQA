import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, GenerationConfig
from MTable import apply_table_llama, apply_table_function
from Utils.dataLoader import TaskCore

class TableQAEvaluator:
    def __init__(self, model_path, device="cuda:0"):
        # 初始化 TableLlama 模型
        self.device = device

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
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        apply_table_llama(
                self.model,
                starting_layer=7,
                ending_layer=12,
                entropy_threshold=0.75,
                retracing_ratio=0.1
            )
        print(f"模型 {model_path} 已加载完成")

    def answer_question(self, db_str, question, choices_str, meta_info=None, prompt_type="default"):
        """
        回答问题
        
        参数:
        - db_str: 数据库表格的字符串表示
        - question: 问题文本
        - choices_str: 选项字符串
        - meta_info: 元信息(可选)
        - prompt_type: 提示类型，可选值为 "default", "cot", "retrace_table"
        
        返回:
        - 模型的回答
        """
        # 根据不同的提示类型构建完整的提示
        if prompt_type == "default":
            # 原始提问方式
            full_prompt = f"{db_str}\n\nPlease carefully analyze and answer the following question step by step.\n\n{question}"
            
        elif prompt_type == "cot":
            # 加入 CoT (Chain-of-Thought) 的提问方式
            full_prompt = f"{db_str}\n\nPlease carefully analyze and answer the following question step by step.\n\n{question}\n\nFollow these steps:\n1. Analyze the table structure and relationships\n2. Identify the tables and fields needed to answer the question\n3. If multiple tables are involved, consider their relationships\n4. Perform necessary data operations (filtering, joining, calculating, etc.)\n5. Derive the final answer"
            
        elif prompt_type == "retrace_table":
            # 针对多表关系的提问方式，配合表格增强功能
            full_prompt = f"{db_str}\n\nPlease carefully analyze and answer the following question step by step.\n\n{question}\n\nThis is a multi-table query. First analyze the relationships between tables (such as foreign key associations), then determine which tables you need to extract information from, and finally derive the answer through table joins and data processing."
        
        else:
            # 默认提问方式
            full_prompt = f"{db_str}\n\nPlease carefully analyze and answer the following question step by step.\n\n{question}"
        
        # 如果有选项，添加到提示中
        if choices_str:
            full_prompt += f"\n\nThis question has only one correct answer. Please break down the question, evaluate each option, and explain why it is correct or incorrect.\n\n{choices_str}\n\nConclude with your final choice on a new line formatted as `Answer: A/B/C/D`."
        
        # 准备输入
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
            truncation=True
        ).to(self.device)

        # 生成回答
        use_table_token = prompt_type == "retrace_table"  # 只在 retrace_table 模式下传入表格内容
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            table_token=db_str if use_table_token else None,  # 只在特定模式下传入表格内容
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
        
        # 运行评估
        task_core.testAll(
            model=model_name,
            dbn=dataset_name,
            scale=scale,
            markdown=markdown,
            dbLimit=db_limit,
            sampleLimit=sample_limit,
            questionLimit=question_limit,
            func=wrapped_answer_func,
            timeSleep=time_sleep
        )
        
        print(f"评估完成，结果已保存到 {result_path}")

def main():
    parser = argparse.ArgumentParser(description="表格问答评估工具")
    parser.add_argument("--model_path", type=str, default="chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct", 
                        help="模型路径")
    parser.add_argument("--db_root", type=str, required=True, help="数据库根目录")
    parser.add_argument("--task_path", type=str, required=True, help="任务文件路径")
    parser.add_argument("--result_path", type=str, required=True, help="结果保存路径")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--scale", type=str, default="small", help="数据规模")
    parser.add_argument("--markdown", action="store_true", help="使用markdown格式")
    parser.add_argument("--db_limit", type=int, default=5, help="数据库数量限制")
    parser.add_argument("--sample_limit", type=int, default=5, help="每个数据库的样本数量限制")
    parser.add_argument("--question_limit", type=int, default=5, help="每个样本的问题数量限制")
    parser.add_argument("--time_sleep", type=float, default=0, help="每次评估间隔时间")
    parser.add_argument("--prompt_type", type=str, default="default", 
                        choices=["default", "cot", "retrace_table"],
                        help="提示类型: default(原始提问), cot(思维链), retrace_table(表格增强)")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = TableQAEvaluator(args.model_path)
    
    # 如果不使用表格增强功能，则禁用它
    if args.prompt_type != "retrace_table" :
        # 禁用表格增强功能的代码
        for layer in evaluator.model.model.layers:
            if hasattr(layer.mlp, 'apply_table_injection'):
                layer.mlp.apply_table_injection = False
    
    # 运行评估
    evaluator.run_evaluation(
        db_root=args.db_root,
        task_path=args.task_path,
        result_path=args.result_path,
        dataset_name=args.dataset,
        scale=args.scale,
        markdown=args.markdown,
        db_limit=args.db_limit,
        sample_limit=args.sample_limit,
        question_limit=args.question_limit,
        time_sleep=args.time_sleep,
        prompt_type=args.prompt_type
    )


# Single question test example
def test_single_question():
    model_path = "chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    evaluator = TableQAEvaluator(model_path)

    # Table content for multi-table association
    table_content = """
    ## employees
    | employee_id | name  | department_id | position       | salary |
    |-------------|-------|---------------|----------------|--------|
    | 1           | Zhang San | 101           | Senior Engineer | 25000  |
    | 2           | Li Si   | 101           | Engineer       | 18000  |
    | 3           | Wang Wu | 102           | Sales Manager  | 20000  |
    | 4           | Zhao Liu| 102           | Sales Representative | 15000  |
    | 5           | Qian Qi | 103           | Finance Supervisor | 22000  |

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
    question = "What is the total budget of the projects managed by the R & D Department? Please provide the answer in the format of \"Answer: X\" in the last line.\nA. 500\nB. 600\nC. 1100\nD. 1650"
    '''
    正确答案
    这些项目的总预算是 1100000
    '''

    # Test three different question - asking methods
    for prompt_type in ["default", "cot", "retrace_table"]:
        print(f"\n===== Question - asking method: {prompt_type} =====")
        response = evaluator.answer_question(table_content, question, "", prompt_type=prompt_type)
        # print("Question:", question)
        print("Answer:", response)

if __name__ == "__main__":
    # If this script is run directly, execute the single question test
    if not any('--' in arg for arg in os.sys.argv[1:]):
        test_single_question()
    else:
        main()