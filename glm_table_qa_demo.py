import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from MTable import apply_table_function_glm,apply_table_llama
from Utils.table_processor_single import SingleTableProcessor

# 设置环境变量和模型路径
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置GPU编号
MODEL_PATH = "GLM/"  # 模型路径

class GLMTableQADemo:
    def __init__(self, model_path, device="cuda"):
        # 初始化设备
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 应用表格增强功能
        apply_table_function_glm()
        
        # 加载分词器
        print(f"正在加载GLM模型分词器: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        print(f"正在加载GLM模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        # 初始化表格处理器
        self.table_token_budget = 8000
        self.table_processor = SingleTableProcessor(self.tokenizer, self.device, self.table_token_budget)
        
        apply_table_llama(
                self.model,
                starting_layer=10,
                ending_layer=13,
                entropy_threshold=0.9,
                retracing_ratio=0.05
            )
        print(f"模型 {model_path} 已加载完成")
    
    def answer_question(self, table_content, question, use_table_token=False):
        """
        使用GLM模型回答表格相关问题
        
        参数:
        - table_content: 表格内容
        - question: 问题
        - use_table_token: 是否使用表格增强功能
        
        返回:
        - 模型回答
        """
        # 构建提示
        prompt = f"请分析以下表格并回答问题:\n\n{table_content}\n\n问题: {question}"
        
        # 为GLM模型准备输入
        messages = [
            {"role": "system", "content": "你是一个擅长分析表格数据的AI助手。"},
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板
        # 修改为使用两步处理方式，先获取文本，再进行tokenize
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 使用tokenizer处理输入，确保返回attention_mask
        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True  # 确保返回attention_mask
        ).to(self.device)
        
        # 处理表格内容（如果启用表格增强）
        table_token_ids = None
        if use_table_token:
            table_token_ids = self.table_processor.process_table_content(table_content, question, False)
            if table_token_ids is not None:
                table_token_ids = table_token_ids.to(self.device)
        
        # 生成配置
        gen_kwargs = {
            "max_new_tokens": 2000,
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
        
        # 打印token数量
        tokens = self.tokenizer.encode(response, add_special_tokens=False)
        print(f"回答的token数量: {len(tokens)}")
        
        return response

def main():
    # 初始化演示类
    demo = GLMTableQADemo(MODEL_PATH)
    
    # 示例表格内容
    table_content = """
    # 员工信息表
    
    ## 员工
    
    员工ID,姓名,部门ID,职位,薪资,入职日期
    1,张三,101,工程师,15000,2020-01-15
    2,李四,102,设计师,12000,2019-05-20
    3,王五,101,高级工程师,20000,2018-03-10
    4,赵六,103,产品经理,18000,2021-02-01
    5,钱七,102,UI设计师,13000,2020-07-15
    
    ## 部门
    
    部门ID,部门名称,部门主管,位置
    101,研发部,张明,A栋3层
    102,设计部,刘芳,A栋2层
    103,产品部,陈强,B栋1层
    
    ## 项目
    
    项目ID,项目名称,负责部门,预算,开始日期,结束日期
    P001,移动应用开发,101,500000,2022-01-01,2022-06-30
    P002,网站重设计,102,300000,2022-02-15,2022-05-15
    P003,新产品规划,103,450000,2022-03-01,2022-08-31
    """
    
    # 示例问题
    questions = [
        "研发部有多少名员工？",
        "哪个部门的平均薪资最高？",
        "所有项目的总预算是多少？"
    ]
    
    # 测试不同问题
    for i, question in enumerate(questions):
        print(f"\n===== 问题 {i+1}: {question} =====")
        
        # 不使用表格增强
        print("\n不使用表格增强:")
        response = demo.answer_question(table_content, question, use_table_token=False)
        print(f"回答: {response}")
        
        # 使用表格增强
        print("\n使用表格增强:")
        response = demo.answer_question(table_content, question, use_table_token=True)
        print(f"回答: {response}")

if __name__ == "__main__":
    main()