import os
import argparse
import time
from openai import OpenAI
from tqdm import tqdm
from Utils.dataLoader import TaskCore
from symbolic import dataDict

class TableLlamaAPI:
    def __init__(self, api_base_url, api_key, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        初始化 TableLlamaAPI
        
        参数:
            api_base_url: API 基础 URL
            api_key: API 密钥
            model_name: 模型名称
        """
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.system_message = "You are a helpful assistant specialized in analyzing tables and answering questions based on tabular data."
    
    def _load_prompt_templates(self, prompt_type="default"):
        """
        加载提示模板
        
        参数:
            prompt_type: 提示类型，可选值为 "default", "cot"
            
        返回:
            提示模板
        """
        prompt_file_path = os.path.join("./prompts", f"{prompt_type}_prompt.txt")
        
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            print(f"警告: 未找到提示文件 {prompt_file_path}，使用默认提示")
            # 如果找不到文件，使用默认提示
            if prompt_type == "default":
                prompt_template = "Please carefully analyze and answer the following question:\n\n{db_str}\n\n{question}\n\nThis question has only one correct answer. Please break down the question, evaluate each option, and explain why it is correct or incorrect. Conclude with your final choice on a new line formatted as `Answer: A/B/C/D`."
            elif prompt_type == "cot":
                prompt_template = "Please carefully analyze and answer the following question step by step:\n\n{db_str}\n\n{question}\n\nThis question has only one correct answer. Please break down the question, evaluate each option, and explain why it is correct or incorrect. Conclude with your final choice on a new line formatted as `Answer: A/B/C/D`."
        
        return prompt_template

    def chat(self, message, temperature=0.7, max_tokens=2048):
        """
        使用 API 进行对话
        
        参数:
            message: 用户消息
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            
        返回:
            模型回复
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': self.system_message
                    },
                    {
                        'role': 'user',
                        'content': message
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 调用出错: {e}")
            return f"发生错误: {str(e)}"

    def answer_question(self, db_str, question, choices_str, meta_info=None, prompt_type="default"):
        """
        回答问题
        
        参数:
            db_str: 数据库表格的字符串表示
            question: 问题文本
            choices_str: 选项字符串
            meta_info: 元信息(可选)
            prompt_type: 提示类型，可选值为 "default", "cot"
            
        返回:
            模型的回答
        """
        # 加载提示模板
        prompt_template = self._load_prompt_templates(prompt_type)
        
        # 构建完整提示
        full_prompt = prompt_template.format(db_str=db_str, question=question)
        
        # 添加选项
        if choices_str and "{choices_str}" not in prompt_template:
            full_prompt += f"\n\n{choices_str}"
        elif choices_str:
            full_prompt = full_prompt.replace("{choices_str}", choices_str)
        
        # 调用 API
        response = self.chat(full_prompt, temperature=0.85)
        
        # 确保回答包含答案格式
        if "answer:" not in response.lower():
            print("警告: 回答中未找到 'Answer:' 格式")
        
        return response

    def run_evaluation(self, db_root, task_path, result_path, 
                      dataset_name, scale, markdown=True, 
                      db_limit=5, sample_limit=5, question_limit=5, 
                      time_sleep=0, prompt_type="default"):
        """
        运行评估
        
        参数:
            db_root: 数据库根目录
            task_path: 任务文件路径
            result_path: 结果保存路径
            dataset_name: 数据集名称
            scale: 数据规模
            markdown: 是否使用markdown格式
            db_limit, sample_limit, question_limit: 评估范围限制
            time_sleep: 每次评估间隔时间
            prompt_type: 提示类型，可选值为 "default", "cot"
        """
        # 初始化TaskCore
        task_core = TaskCore(db_root, task_path, result_path)
        
        # 获取模型名称，根据提示类型添加后缀
        model_name = f"TableLlamaAPI_{prompt_type}"
        
        # 创建一个包装函数，将prompt_type传递给answer_question
        def wrapped_answer_func(db_str, question, choices_str, meta_info=None):
            return self.answer_question(db_str, question, choices_str, meta_info, prompt_type=prompt_type)
        
        # 获取数据库列表
        database_list = list(dataDict.keys())
        
        for dbn in tqdm(database_list, desc="database_list"):
            # 处理scale参数
            if isinstance(scale, list):
                scale_list = scale
            else:
                scale_list = [scale]
            
            for current_scale in scale_list:
                # 根据不同规模设置等待时间
                current_time_sleep = time_sleep
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
    parser = argparse.ArgumentParser(description="多表格问答评估")
    parser.add_argument("--api_base_url", type=str, default="https://api-inference.modelscope.cn/v1/", 
                        help="API基础URL")
    # parser.add_argument("--api_key", type=str, required=True, 
    #                     help="API密钥")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                        help="模型名称")
    parser.add_argument("--db_root", type=str, required=True, 
                        help="数据库根目录")
    parser.add_argument("--task_path", type=str, required=True, 
                        help="任务文件路径")
    parser.add_argument("--result_path", type=str, required=True, 
                        help="结果保存路径")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="数据集名称")
    parser.add_argument("--scale", type=str, nargs='+', default=["8k"], 
                        choices=["8k", "16k", "32k", "64k", "128k"],
                        help="数据规模，可指定多个值，如: 8k 16k 32k")
    parser.add_argument("--markdown", action="store_true", 
                        help="使用markdown格式")
    parser.add_argument("--db_limit", type=int, default=5, 
                        help="数据库数量限制")
    parser.add_argument("--sample_limit", type=int, default=5, 
                        help="每个数据库的样本数量限制")
    parser.add_argument("--question_limit", type=int, default=5, 
                        help="每个样本的问题数量限制")
    parser.add_argument("--time_sleep", type=float, default=0, 
                        help="每次评估间隔时间")
    parser.add_argument("--prompt_type", type=str, default="default", 
                        choices=["default", "cot"],
                        help="提示类型: default(原始提问), cot(思维链)")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = TableLlamaAPI(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name
    )
    
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


# 添加单个问题测试示例
def test_single_question():
    # 初始化API客户端
    api_base_url = "https://api-inference.modelscope.cn/v1/"
    api_key = "ee92f5a9-4138-4235-81c7-e1f6cb8c23ca"  # 请替换为您的API密钥
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    evaluator = TableLlamaAPI(api_base_url, api_key, model_name)

    # 表格内容
    table_content = """
    #airline

    ## Air_Carriers

    Code,Description
    19393,Southwest Airlines Co.: WN
    19687,Horizon Air: QX
    19790,Delta Air Lines Inc.: DL
    19805,American Airlines Inc.: AA
    19930,Alaska Airlines Inc.: AS
    19977,United Air Lines Inc.: UA
    20046,Air Wisconsin Airlines Corp: ZW
    20304,SkyWest Airlines Inc.: OO
    20363,Endeavor Air Inc.: 9E
    20368,Allegiant Air: G4
    20378,Mesa Airlines Inc.: YV
    20397,PSA Airlines Inc.: OH
    20398,Envoy Air: MQ
    20409,JetBlue Airways: B6
    20416,Spirit Air Lines: NK
    20452,Republic Airline: YX


    ## Airports

    Code,Description
    ABQ,"Albuquerque, NM: Albuquerque International Sunport"
    ATL,"Atlanta, GA: Hartsfield-Jackson Atlanta International"
    BUF,"Buffalo, NY: Buffalo Niagara International"
    BWI,"Baltimore, MD: Baltimore/Washington International Thurgood Marshall"
    CLT,"Charlotte, NC: Charlotte Douglas International"
    CVG,"Cincinnati, OH: Cincinnati/Northern Kentucky International"
    DAL,"Dallas, TX: Dallas Love Field"
    DTW,"Detroit, MI: Detroit Metro Wayne County"
    FLL,"Fort Lauderdale, FL: Fort Lauderdale-Hollywood International"
    FNT,"Flint, MI: Bishop International"
    """

    # 问题
    question = "How many airlines land in Buffalo, NY: Buffalo Niagara International?:\n A. 1 \nB. 2\nC. 0 \nD. 3"

    # 测试不同提示类型
    for prompt_type in ["default", "cot"]:
        print(f"\n===== 提示类型: {prompt_type} =====")
        response = evaluator.answer_question(table_content, question, "", prompt_type=prompt_type)
        print("回答:", response)


if __name__ == "__main__":
    # 如果直接运行脚本且没有命令行参数，则执行单个问题测试
    if len(os.sys.argv) <= 1:
        test_single_question()
    else:
        main()