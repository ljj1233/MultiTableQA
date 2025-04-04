import os
import re
import argparse
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import time
import openai
from Utils.jsTool import JS
from eval.evaluator import evalAcc

class APIModelEvaluator:
    def __init__(self, model_name, api_key=None):
        """
        初始化API模型评估器
        
        Args:
            model_name: 模型名称 (如 'gpt-4o', 'claude-3')
            api_key: API密钥 (可选，也可从环境变量获取)
        """
        self.model_name = model_name
        
        # 设置API密钥
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.environ.get("OPENAI_API_KEY") and model_name.startswith("gpt-"):
            raise ValueError("请设置OPENAI_API_KEY环境变量或通过api_key参数提供")
            
        # 加载提示模板
        self.prompt_templates = self._load_prompt_templates()
        
        print(f"API模型 {model_name} 已准备就绪")
        
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
        os.makedirs("d:/NLP/MultiTableQA/prompts", exist_ok=True)
        
        # 加载提示模板
        for prompt_type, file_path in template_files.items():
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    templates[prompt_type] = f.read().strip()
            else:
                print(f"警告: 提示模板文件 {file_path} 不存在")
        
        return templates

    def answer_question(self, db_str, question, choices_str, meta_info=None, prompt_type="default", request_id=None, log_root=None):
        """
        调用API模型回答问题
        
        Args:
            db_str: 数据库表格的字符串表示
            question: 问题文本
            choices_str: 选项字符串
            meta_info: 元信息(可选)
            prompt_type: 提示类型
            request_id: 请求ID (用于日志)
            log_root: 日志根目录
            
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
        
        # 记录请求
        if log_root and request_id:
            os.makedirs(log_root, exist_ok=True)
            log_file = os.path.join(log_root, f"{request_id}.txt")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Prompt:\n{full_prompt}\n\n")
        
        # 调用API
        if self.model_name.startswith("gpt-"):
            # OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0
            )
            answer = response.choices[0].message.content
        elif self.model_name.startswith("claude-"):
            # Anthropic Claude API (需要安装anthropic包)
            try:
                import anthropic
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    temperature=0,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                answer = response.content[0].text
            except ImportError:
                raise ImportError("使用Claude模型需要安装anthropic包: pip install anthropic")
        else:
            raise ValueError(f"不支持的模型类型: {self.model_name}")
            
        # 记录回答
        if log_root and request_id:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"Response:\n{answer}")
                
        return answer

def main():
    parser = argparse.ArgumentParser(description="API表格问答模型评估工具")
    parser.add_argument("--model", type=str, required=True, help="模型名称 (如 gpt-4o, claude-3)")
    parser.add_argument("--dataset", type=str, required=True, help="数据集类型 (qa, fv, ret, cpa, cta, em, bqa)")
    parser.add_argument("--scale", type=str, default="16k", help="数据规模 (8k, 16k, 32k, 64k, 128k)")
    parser.add_argument("--markdown", action="store_true", help="使用markdown格式")
    parser.add_argument("--prompt_type", type=str, default="default", 
                        choices=["default", "cot", "retrace_table"],
                        help="提示类型: default(原始提问), cot(思维链), retrace_table(表格增强)")
    parser.add_argument("--log_root", type=str, default=None, help="日志根目录")
    parser.add_argument("--result_path", type=str, default=None, help="结果JSON路径")
    parser.add_argument("--api_key", type=str, default=None, help="API密钥")
    
    args = parser.parse_args()
    
    # 初始化API模型评估器
    evaluator = APIModelEvaluator(args.model, args.api_key)
    
    # 运行评估
    evalAcc(
        ds=args.dataset,
        scale=args.scale,
        markdown=args.markdown,
        model=args.model,
        evaluator=evaluator,
        prompt_type=args.prompt_type,
        logRoot=args.log_root,
        resultPath=args.result_path,
        is_api_model=True
    )

# 单个问题测试示例
def test_single_question():
    model_name = "gpt-4o"
    evaluator = APIModelEvaluator(model_name)
    
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
    """
    
    # 多表关联的问题
    question = "研发部负责了哪些项目？这些项目的总预算是多少？"
    
    # 测试三种不同的提问方式
    for prompt_type in ["default", "cot", "retrace_table"]:
        print(f"\n===== 提问方式: {prompt_type} =====")
        response = evaluator.answer_question(table_content, question, "", prompt_type=prompt_type)
        print("问题:", question)
        print("回答:", response)

if __name__ == "__main__":
    # 如果直接运行此脚本，则执行单个问题测试
    if not any('--' in arg for arg in os.sys.argv[1:]):
        test_single_question()
    else:
        main()