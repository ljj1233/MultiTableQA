import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, GenerationConfig
from multi_Table import apply_table_llama, apply_table_function

class TableLlama:
    def __init__(self, model_path: str, device: str = "cuda:0"):
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
            starting_layer=5,
            ending_layer=7,
            entropy_threshold=0.5,
            retracing_ratio=0.05
        )

    def chat(self, message: str, table_token: str = None):
        messages = [{"role": "user", "content": message}]
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

        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            table_token=table_token,
            tokenizer=self.tokenizer
        )

        response = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return response

    def answer_multiple_choice(self, db_str, question, choices_str, meta_info=None):
        """
        回答多选题问题，与TaskCore兼容的接口
        
        参数:
        - db_str: 数据库表格的字符串表示
        - question: 问题文本
        - choices_str: 选项字符串
        - meta_info: 元信息(可选)，包含(dbn, scale, dbIdx, sampleIdx, questionIdx, rightIdx)
        
        返回:
        - 模型的回答，包含"Answer: X"格式的答案
        """
        # 构建完整的提示，使用英文并参考提供的模板
        full_prompt = f"{db_str}\n\nPlease carefully analyze and answer the following single choice question step by step.\n\n{question}\n\nThis question has only one correct answer. Please break down the question, evaluate each option, and explain why it is correct or incorrect.\n\n{choices_str}\n\nConclude with your final choice on a new line formatted as `Answer: A/B/C/D`."
        
        # 使用现有的chat方法
        response = self.chat(full_prompt, table_token=db_str)
        
        # 确保回答包含答案格式
        if "answer:" not in response.lower():
            response += "\n\nAnswer: "   
            
        return response

def main():
    # 初始化模型
    model_path = "chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct"  
    llm = TableLlama(model_path)
    
    # 多表关联的表格内容
    table_content = """
    ## employees
    | employee_id | name  | department_id | position       | salary |
    |-------------|-------|---------------|----------------|--------|
    | 1           | 张三  | 101           | 高级工程师     | 25000  |
    | 2           | 李四  | 101           | 工程师         | 18000  |
    | 3           | 王五  | 102           | 销售经理       | 20000  |
    | 4           | 赵六  | 102           | 销售代表       | 15000  |
    | 5           | 钱七  | 103           | 财务主管       | 22000  |
    
    ## departments
    | department_id | department_name | location    | manager_id |
    |---------------|----------------|-------------|------------|
    | 101           | 研发部         | 北京        | 1          |
    | 102           | 销售部         | 上海        | 3          |
    | 103           | 财务部         | 广州        | 5          |
    
    ## projects
    | project_id | project_name | department_id | start_date  | end_date    | budget  |
    |------------|--------------|---------------|-------------|-------------|---------|
    | 201        | 产品A开发    | 101           | 2023-01-15  | 2023-06-30  | 500000  |
    | 202        | 市场推广     | 102           | 2023-02-01  | 2023-04-30  | 300000  |
    | 203        | 财务系统升级 | 103           | 2023-03-10  | 2023-05-15  | 250000  |
    | 204        | 产品B开发    | 101           | 2023-04-01  | 2023-09-30  | 600000  |
    
    ## assignments
    | assignment_id | employee_id | project_id | role          | hours_allocated |
    |---------------|-------------|------------|---------------|-----------------|
    | 301           | 1           | 201        | 项目负责人    | 160             |
    | 302           | 2           | 201        | 开发人员      | 120             |
    | 303           | 1           | 204        | 技术顾问      | 80              |
    | 304           | 2           | 204        | 开发人员      | 140             |
    | 305           | 3           | 202        | 项目负责人    | 100             |
    | 306           | 4           | 202        | 销售支持      | 120             |
    | 307           | 5           | 203        | 项目负责人    | 90              |
    """
    
    # 多表关联的查询问题
    question = "研发部有哪些员工？他们参与了哪些项目，担任什么角色？"
    response = llm.chat(question, table_content)
    print("问题:", question)
    print("回答:", response)
    

    question2 = "哪个部门的项目预算总和最高？该部门的经理是谁？"
    response2 = llm.chat(question2, table_content)
    print("\n问题:", question2)
    print("回答:", response2)

# 添加与TaskCore集成的示例
def evaluate_with_taskcore(model_path, db_root, task_path, result_path, dataset, scale="small", markdown=True):
    """
    使用TaskCore评估TableLlama模型
    
    参数:
    - model_path: 模型路径
    - db_root: 数据库根目录
    - task_path: 任务文件路径
    - result_path: 结果保存路径
    - dataset: 数据集名称
    - scale: 数据规模
    - markdown: 是否使用markdown格式
    """
    from Utils.dataLoader import TaskCore
    
    # 初始化模型
    llm = TableLlama(model_path)
    
    # 初始化TaskCore
    task_core = TaskCore(db_root, task_path, result_path)
    
    # 运行评估
    task_core.testAll(
        model="TableLlama",
        dbn=dataset,
        scale=scale,
        markdown=markdown,
        dbLimit=5,  # 可以根据需要调整
        sampleLimit=5,
        questionLimit=5,
        func=llm.answer_multiple_choice,
        timeSleep=0
    )
    
    print(f"评估完成，结果已保存到 {result_path}")

if __name__ == "__main__":
    main()