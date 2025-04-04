import os
import torch
import argparse
import re
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, GenerationConfig
from multi_Table import apply_table_llama, apply_table_function
from Utils.jsTool import JS

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
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 应用表格注入增强
        apply_table_llama(
            self.model,
            starting_layer=5,
            ending_layer=7,
            entropy_threshold=0.5,
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
            "default": "d:/NLP/MultiTableQA/prompts/default_prompt.txt",
            "cot": "d:/NLP/MultiTableQA/prompts/cot_prompt.txt",
            "retrace_table": "d:/NLP/MultiTableQA/prompts/retrace_table_prompt.txt"
        }
        
        # 确保提示目录存在
        os.makedirs("d:/NLP/MultiTableQA/prompts", exist_ok=True)
        
        # 加载或创建默认提示模板
        for prompt_type, file_path in template_files.items():
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    templates[prompt_type] = f.read().strip()
            else:
                # 创建默认模板
                if prompt_type == "default":
                    default_template = "{db_str}\n\nPlease carefully analyze and answer the following question step by step.\n\n{question}"
                elif prompt_type == "cot":
                    default_template = "{db_str}\n\nPlease carefully analyze and answer the following question step by step.\n\n{question}\n\nFollow these steps:\n1. Analyze the table structure and relationships\n2. Identify the tables and fields needed to answer the question\n3. If multiple tables are involved, consider their relationships\n4. Perform necessary data operations (filtering, joining, calculating, etc.)\n5. Derive the final answer"
                elif prompt_type == "retrace_table":
                    default_template = "{db_str}\n\nPlease carefully analyze and answer the following question step by step.\n\n{question}\n\nThis is a multi-table query. First analyze the relationships between tables (such as foreign key associations), then determine which tables you need to extract information from, and finally derive the answer through table joins and data processing."
                
                # 保存默认模板
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(default_template)
                templates[prompt_type] = default_template
                
        return templates

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

def extractAnswer(text:str)->str:
    """
    从模型输出中提取单个答案
    
    Args:
        text: 模型输出文本
        
    Returns:
        提取的答案（A-F）
    """
    patt = r'answer:\s*([A-F]+)'
    grps = re.findall(patt, text, re.IGNORECASE)
    if grps:
        return grps[-1].upper()
    return ''

def extractBatchedAnswer(idx:int, text:str)->str:
    """
    从模型输出中提取批处理中的特定答案
    
    Args:
        idx: 问题索引
        text: 模型输出文本
        
    Returns:
        提取的答案（A-F）
    """
    patt = rf'answer\s*{idx}:\s*([A-F]+)'
    grps = re.findall(patt, text, re.IGNORECASE)
    if grps:
        return grps[-1].upper()
    return ''

def evalFile(filePath):
    """
    评估结果文件
    
    Args:
        filePath: 结果文件路径
    """
    saveList = JS(filePath).loadJS()
    cnt = sum([1 for item in saveList if item['right']])
    err = sum([1 for item in saveList if item['error'] is not None])
    tot = len(saveList)
    print('right choices', cnt)
    print('call errors', err)
    print('total', tot)
    print('acc (ignore call errors)', cnt / (tot - err))
    print('acc', cnt / tot)

def evalAcc(ds,       # 数据集类型
            scale,    # 数据规模 8k-128k (不适用于em)
            markdown, # 是否使用markdown格式 (不适用于em)
            model_path, # 模型路径
            prompt_type="default", # 提示类型
            logRoot=None,  # 日志根目录
            resultPath=None # 结果JSON路径
            ):
    """
    评估模型在特定数据集上的准确率
    
    Args:
        ds: 数据集类型
        scale: 数据规模
        markdown: 是否使用markdown格式
        model_path: 模型路径
        prompt_type: 提示类型
        logRoot: 日志根目录
        resultPath: 结果JSON路径
    """
    # 导入数据集类型
    from benchmarkLoader.tableQALoader import TableQADataset
    from benchmarkLoader.tableFVLoader import TableFVDataset
    from benchmarkLoader.retrievalLoader import RetrievalDataset
    from benchmarkLoader.cpaLoader import CPADataset
    from benchmarkLoader.ctaLoader import CTADataset
    from benchmarkLoader.emLoader import EMDataset
    from benchmarkLoader.batchedTableQALoader import BatchedTableQADataset
    
    # 数据集类型映射字典
    dsDict = {
        'qa': TableQADataset,    # 表格问答
        'fv': TableFVDataset,    # 表格事实验证
        'ret': RetrievalDataset, # 表格检索
        'cpa': CPADataset,       # 列属性预测
        'cta': CTADataset,       # 列类型分析
        'em': EMDataset,         # 实体匹配
        'bqa': BatchedTableQADataset  # 批处理表格问答
    }
    
    if ds not in dsDict.keys():
        print(f"未知数据集类型: {ds}")
        return None
        
    # 设置默认日志目录
    if logRoot == None:
        logRoot = os.path.join('results', ds)
        
    # 设置默认结果路径
    if resultPath == None:
        tmp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + "_" + str(uuid4()) + ".json"
        model_name = os.path.basename(model_path.rstrip("/\\"))
        resultName = f'{ds}_{scale}_{markdown}_{model_name}_{prompt_type}_{tmp}'
        resultPath = os.path.join('results', resultName)

    # 初始化数据集
    dataset = None
    if ds == 'em':
        dataset = dsDict[ds]()
    elif ds.startswith('b'):
        # 批处理输入的情况
        dataset = dsDict[ds](4, scale, markdown)
    else:
        dataset = dsDict[ds](scale, markdown)

    # 初始化模型
    evaluator = TableLlamaEvaluator(model_path)
    
    # 如果不使用表格增强功能，则禁用它
    if prompt_type != "retrace_table":
        # 禁用表格增强功能的代码
        for layer in evaluator.model.model.layers:
            if hasattr(layer.mlp, 'apply_table_injection'):
                layer.mlp.apply_table_injection = False
    
    idx = 0
    saveList = []
    
    # 处理批处理数据集
    if ds.startswith('b'):
        for q, c in tqdm(dataset, desc=ds):
            pred = ['' for _ in range(len(c))]
            err = None
            try:
                # 调用模型
                res = evaluator.answer_question(
                    q,
                    "",
                    "",
                    None,
                    prompt_type
                )
                # 提取每个问题的答案
                for i in range(len(c)):
                    pred[i] = extractBatchedAnswer(i, res)
            except Exception as e:
                err = str(e)
                
            # 保存每个问题的结果
            for i in range(len(c)):
                saveList.append({
                    'idx': idx,
                    'gt': c[i],
                    'pred': pred[i],
                    'right': c[i] == pred[i],
                    'error': err
                })
                JS(resultPath).newJS(saveList)
                idx += 1
    # 处理普通数据集
    else:
        for q, c in tqdm(dataset, desc=ds):
            pred = ''
            err = None
            try:
                # 调用模型
                res = evaluator.answer_question(
                    q,
                    "",
                    "",
                    None,
                    prompt_type
                )
                # 提取答案
                pred = extractAnswer(res)
            except Exception as e:
                err = str(e)
                
            # 保存结果
            saveList.append({
                'idx': idx,
                'gt': c,
                'pred': pred,
                'right': c == pred,
                'error': err
            })
            JS(resultPath).newJS(saveList)
            idx += 1

    # 评估结果
    evalFile(resultPath)

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
    
    # 运行评估
    evalAcc(
        ds=args.dataset,
        scale=args.scale,
        markdown=args.markdown,
        model_path=args.model_path,
        prompt_type=args.prompt_type,
        logRoot=args.log_root,
        resultPath=args.result_path
    )

# 单个问题测试示例
def test_single_question():
    model_path = "chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    evaluator = TableLlamaEvaluator(model_path)
    
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