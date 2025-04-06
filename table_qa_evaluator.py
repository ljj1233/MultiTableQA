import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, GenerationConfig
from MTable import apply_table_llama, apply_table_function
from Utils.dataLoader import TaskCore
from symbolic import dataDict

from tqdm import tqdm
class TableQAEvaluator:
    def __init__(self, model_path, device="cuda:0", multi_gpu=False):
        # 初始化 TableLlama 模型
        self.device = device
        self.multi_gpu = multi_gpu

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
            max_length=512,
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
        # 根据提示类型读取对应的prompt文件
        prompt_file_path = os.path.join("./prompts", f"{prompt_type}_prompt.txt")
        
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            print(f"警告: 未找到提示文件 {prompt_file_path}，使用默认提示")
            # 如果找不到文件，使用默认提示
            prompt_template = "Please carefully analyze and answer the following question:\n\n{db_str}\n\n{question}\n\nThis question has only one correct answer. Please break down the question, evaluate each option, and explain why it is correct or incorrect. Conclude with your final choice on a new line formatted as `Answer: A/B/C/D`."
        
        # 替换模板中的占位符
        full_prompt = prompt_template.format(db_str=db_str,question=question)
        print(f'full_prompt: {full_prompt}')
        # 如果有选项且模板中没有包含选项的占位符，则添加选项
        if choices_str and "{choices_str}" not in prompt_template:
            full_prompt += f"\n\n{choices_str}"
        elif choices_str:
            full_prompt = full_prompt.replace("{choices_str}", choices_str)
        
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
            max_new_tokens=800,
            temperature=0.85,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.0,
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
    
    args = parser.parse_args()
    
    # 初始化评估器，传入设备和多GPU参数
    evaluator = TableQAEvaluator(args.model_path, device=args.device, multi_gpu=args.multi_gpu)
    
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