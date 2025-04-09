import os
import torch
import argparse
import re
import sqlite3  
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, GenerationConfig
from modelscope import AutoModelForCausalLM as MSAutoModelForCausalLM, AutoTokenizer as MSAutoTokenizer
from MTable import apply_table_llama, apply_table_function,apply_table_function_qwen,apply_table_function_glm
from Utils.dataLoader import TaskCore
from Utils.table_relevance import TableRelevanceExtractor
from Utils.table_parser import parse_markdown_table  
from Utils.table_processor import TableProcessor  
from Utils.table_processor_single import SingleTableProcessor  

from symbolic import dataDict
import pandas as pd
from io import StringIO

from tqdm import tqdm
class TableQAEvaluator:
    def __init__(self, model_path, device="cuda:0", multi_gpu=False, use_llm_for_relevance=False):
        # 初始化 TableLlama 模型
        self.device = device
        self.multi_gpu = multi_gpu
        self.use_llm_for_relevance = use_llm_for_relevance
        self.table_token_budget = 8000   
        
        apply_table_function_glm()

        # 判断是否为GLM模型
        self.is_glm_model = "glm" in model_path.lower()
        
        # 加载模型和分词器
        print(f"正在加载GLM模型: {model_path}")
        # 使用modelscope加载GLM模型
        self.tokenizer = MSAutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
        )
            
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU 进行并行计算")
            self.model = MSAutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto"  # 自动分配到可用的GPU上
                )
        else:
            self.model = MSAutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to(device).eval()
            
        # GLM模型的生成配置
        self.generation_config = {
                "max_length": 30000,
                "do_sample": True,
                "top_k": 5,
                "top_p": 0.5,
                "temperature": 0.1,
                "repetition_penalty": 1.2,
            }
    
            
        
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
        # 初始化生成配置
        self.generation_config = GenerationConfig(
                num_beams=5,
                max_length=30000,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                # 添加以下参数来控制生成质量
                repetition_penalty=1.2,  # 增加重复惩罚，范围通常在1.0-1.5
                no_repeat_ngram_size=3,  # 禁止重复的n-gram大小
                length_penalty=1.0,  # 长度惩罚，小于1会倾向生成更短的回答
                temperature=0.1,  # 控制生成的随机性，越小越保守
                top_p=0.5,  # 控制采样范围，越小生成越保守
                do_sample=True,
            )

        # 应用表格增强功能
        apply_table_llama(
            self.model,
            starting_layer=10,
            ending_layer=13,
            entropy_threshold=0.9,
            retracing_ratio=0.05
        )
        print(f"模型 {model_path} 已加载完成")
        
        # 初始化表格处理器
        self.table_processor = SingleTableProcessor(self.tokenizer, self.device, self.table_token_budget)


    def _load_prompt_templates(self,prompt_type="default"):
        """
        加载提示模板
        
        Returns:
            提示模板
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
        """
        回答问题 (Modified for GLM model)
        """
        prompt_template = self._load_prompt_templates(prompt_type)
        full_prompt = prompt_template.format(db_str=db_str, question=question) # Use original db_str in prompt
    
        if choices_str and "{choices_str}" not in prompt_template:
            full_prompt += f"\n\n{choices_str}"
        elif choices_str:
            full_prompt = full_prompt.replace("{choices_str}", choices_str)
    
        # 为GLM模型准备输入
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that analyzes tables"
            },
            {
                "role": "user",
                "content": full_prompt 
            }
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
        ).to(self.device)
        
        # --- Generate Table Token IDs for Injection ---
        table_token_ids = None
        use_table_token = prompt_type == "retrace_table"
        if use_table_token:
            # 使用表格处理器处理表格内容
            table_token_ids = self.table_processor.process_table_content(db_str, question, self.use_llm_for_relevance)
    
            if table_token_ids is not None:
                table_token_ids = table_token_ids.to(self.device)
    
        # --- 使用GLM模型生成回答 ---
        gen_kwargs = {
            **self.generation_config,
            "max_new_tokens": 20000,
            "table_token": table_token_ids if use_table_token else None,
            "tokenizer": self.tokenizer if use_table_token else None
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        tokens = self.tokenizer.encode(response, add_special_tokens=False)
        print(f"response 的 token 数量: {len(tokens)}")

        return response


def main():
    parser = argparse.ArgumentParser(description="多表格问答评估")
    parser.add_argument("--model_path", type=str, default="/hpc2hdd/home/fye374/models/Meta-Llama-3.1-8B-Instruct", 
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
    
    # 初始化评估器，传入设备和多GPU参数
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
        
# Single question test example
def test_single_question():
    model_path = "/hpc2hdd/home/fye374/models/Meta-Llama-3.1-8B-Instruct"
    # 检测是否有多个GPU可用
    multi_gpu = torch.cuda.device_count() > 1
    evaluator = TableQAEvaluator(model_path, multi_gpu=multi_gpu)

    # Table content for multi-table association
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
    GEG,"Spokane, WA: Spokane International"
    GSO,"Greensboro/High Point, NC: Piedmont Triad International"
    IAD,"Washington, DC: Washington Dulles International"
    IAH,"Houston, TX: George Bush Intercontinental/Houston"
    IND,"Indianapolis, IN: Indianapolis International"
    JAX,"Jacksonville, FL: Jacksonville International"
    LAS,"Las Vegas, NV: McCarran International"
    LGA,"New York, NY: LaGuardia"
    MCO,"Orlando, FL: Orlando International"
    MEM,"Memphis, TN: Memphis International"
    MHT,"Manchester, NH: Manchester-Boston Regional"
    MKE,"Milwaukee, WI: General Mitchell International"
    MSN,"Madison, WI: Dane County Regional-Truax Field"
    MYR,"Myrtle Beach, SC: Myrtle Beach International"
    OAJ,"Jacksonville/Camp Lejeune, NC: Albert J Ellis"
    OAK,"Oakland, CA: Metropolitan Oakland International"
    ORD,"Chicago, IL: Chicago O'Hare International"
    PDX,"Portland, OR: Portland International"
    RDU,"Raleigh/Durham, NC: Raleigh-Durham International"
    SFO,"San Francisco, CA: San Francisco International"
    SJC,"San Jose, CA: Norman Y. Mineta San Jose International"
    ANC,"Anchorage, AK: Ted Stevens Anchorage International"
    BDL,"Hartford, CT: Bradley International"
    BUR,"Burbank, CA: Bob Hope"
    FAY,"Fayetteville, NC: Fayetteville Regional/Grannis Field"
    JFK,"New York, NY: John F. Kennedy International"
    LBB,"Lubbock, TX: Lubbock Preston Smith International"
    MAF,"Midland/Odessa, TX: Midland International Air and Space Port"
    PGV,"Greenville, NC: Pitt Greenville"
    PHL,"Philadelphia, PA: Philadelphia International"
    PHX,"Phoenix, AZ: Phoenix Sky Harbor International"
    PIE,"St. Petersburg, FL: St Pete Clearwater International"
    SEA,"Seattle, WA: Seattle/Tacoma International"
    VPS,"Valparaiso, FL: Eglin AFB Destin Fort Walton Beach"


    ## Airlines

    FL_DATE,OP_CARRIER_AIRLINE_ID,TAIL_NUM,OP_CARRIER_FL_NUM,ORIGIN_AIRPORT_ID,ORIGIN_AIRPORT_SEQ_ID,ORIGIN_CITY_MARKET_ID,ORIGIN,DEST_AIRPORT_ID,DEST_AIRPORT_SEQ_ID,DEST_CITY_MARKET_ID,DEST,CRS_DEP_TIME,DEP_TIME,DEP_DELAY,DEP_DELAY_NEW,ARR_TIME,ARR_DELAY,ARR_DELAY_NEW,CANCELLED,CANCELLATION_CODE,CRS_ELAPSED_TIME,ACTUAL_ELAPSED_TIME,CARRIER_DELAY,WEATHER_DELAY,NAS_DELAY,SECURITY_DELAY,LATE_AIRCRAFT_DELAY
    2018/8/1,20398,N663AR,3558,13930,1393006,30977,ORD,11721,1172105,31721,FNT,1140,1131.0,-9.0,0.0,1324.0,-15.0,0.0,0,,59,53.0,,,,,,,,,,,,,
    2018/8/1,20378,N86324,6222,15624,1562404,31504,VPS,12266,1226603,31453,IAH,900,854.0,-6.0,0.0,1059.0,11.0,11.0,0,,108,125.0,,,,,,,,
    2018/8/2,19393,N8511K,738,10821,1082106,30852,BWI,11697,1169706,32467,FLL,1945,,,,,,,1,B,155,,,,,,
    2018/8/4,19393,N438WN,5784,13204,1320402,31454,MCO,12339,1233904,32337,IND,2025,2036.0,11.0,11.0,2244.0,-1.0,0.0,0,,140,128.0,,,,,,,
    2018/8/5,20452,N857RW,3629,13930,1393006,30977,ORD,11193,1119302,33105,CVG,910,905.0,-5.0,0.0,1116.0,-13.0,0.0,0,,79,71.0,,,,,,,
    2018/8/5,20409,N947JB,577,11697,1169706,32467,FLL,14771,1477104,32457,SFO,906,940.0,34.0,34.0,1241.0,36.0,36.0,0,,359,361.0,34.0,0.0,2.0,0.0,0.0
    2018/8/7,19393,N7724A,945,14492,1449202,34492,RDU,10821,1082106,30852,BWI,1030,1025.0,-5.0,0.0,1132.0,-3.0,0.0,0,,65,67.0,,,,,,
    2018/8/8,19393,N276WN,2545,13158,1315805,33158,MAF,11259,1125903,30194,DAL,600,557.0,-3.0,0.0,703.0,-7.0,0.0,0,,70,66.0,,,,,,
    2018/8/10,20397,N505AE,5690,11057,1105703,31057,CLT,13485,1348502,33485,MSN,1427,1424.0,-3.0,0.0,1515.0,-18.0,0.0,0,,126,111.0,,,,,,
    2018/8/10,19687,N445QX,2368,14747,1474703,30559,SEA,11884,1188402,31884,GEG,725,724.0,-1.0,0.0,823.0,1.0,1.0,0,,57,59.0,,,,,,
    2018/8/12,20363,N313PQ,4058,10397,1039707,30397,ATL,13795,1379502,33795,OAJ,2116,2112.0,-4.0,0.0,2235.0,-10.0,0.0,0,,89,83.0,,,,,
    2018/8/15,19977,N33266,2103,12266,1226603,31453,IAH,13204,1320402,31454,MCO,2003,2000.0,-3.0,0.0,2310.0,-15.0,0.0,0,,142,130.0,,,,,
    2018/8/15,20304,N814SK,4709,10529,1052906,30529,BDL,14492,1449202,34492,RDU,630,626.0,-4.0,0.0,801.0,-17.0,0.0,0,,108,95.0,,,,,
    2018/8/16,19393,N440LV,1672,10821,1082106,30852,BWI,13296,1329604,30721,MHT,730,724.0,-6.0,0.0,832.0,-18.0,0.0,0,,80,68.0,,,,,
    2018/8/16,20363,N197PQ,3311,10792,1079206,30792,BUF,12953,1295304,31703,LGA,1219,1215.0,-4.0,0.0,1324.0,-16.0,0.0,0,,81,69.0,,,,,
    2018/8/17,20409,N651JB,47,10299,1029906,30299,ANC,14057,1405702,34057,PDX,2359,2349.0,-10.0,0.0,411.0,-23.0,0.0,0,,215,202.0,,,,,
    2018/8/17,19805,N986NN,2310,13930,1393006,30977,ORD,14831,1483106,32457,SJC,1725,1725.0,0.0,0.0,1959.0,5.0,5.0,0,,269,274.0,,,,,
    2018/8/19,20368,302NV,938,14112,1411206,33195,PIE,13342,1334207,33342,MKE,740,735.0,-5.0,0.0,921.0,-2.0,0.0,0,,163,166.0,,,,,
    2018/8/19,20046,N424AW,3922,11641,1164102,31641,FAY,12264,1226402,30852,IAD,1930,1937.0,7.0,7.0,2046.0,-9.0,0.0,0,,85,69.0,,,,,
    2018/8/22,20363,N133EV,5140,12478,1247805,31703,JFK,12451,1245102,31136,JAX,815,809.0,-6.0,0.0,1056.0,3.0,3.0,0,,158,167.0,,,,,
    2018/8/22,20452,N132HQ,4646,14100,1410005,34100,PHL,13244,1324402,33244,MEM,835,831.0,-4.0,0.0,1019.0,-3.0,0.0,0,,167,168.0,,,,,
    2018/8/23,19930,N531AS,796,14747,1474703,30559,SEA,11433,1143302,31295,DTW,2355,25.0,30.0,30.0,738.0,32.0,32.0,0,,251,253.0,30.0,0.0,2.0,0.0,0.0
    2018/8/23,19790,N960DL,942,10397,1039707,30397,ATL,11995,1199502,31995,GSO,816,814.0,-2.0,0.0,913.0,-17.0,0.0,0,,74,59.0,,,,,
    2018/8/24,19393,N733SA,1907,12889,1288903,32211,LAS,10140,1014005,30140,ABQ,2105,2100.0,-5.0,0.0,2323.0,-12.0,0.0,0,,90,83.0,,,,,
    2018/8/24,19393,N8544Z,1564,14107,1410702,30466,PHX,10792,1079206,30792,BUF,1715,1713.0,-2.0,0.0,1.0,-14.0,0.0,0,,240,228.0,,,,,
    2018/8/25,19790,N337DN,1546,14107,1410702,30466,PHX,10397,1039707,30397,ATL,1015,1011.0,-4.0,0.0,1642.0,-23.0,0.0,0,,230,211.0,,,,,
    2018/8/25,20416,N517NK,717,14100,1410005,34100,PHL,13577,1357702,31135,MYR,815,801.0,-14.0,0.0,933.0,-17.0,0.0,0,,95,92.0,,,,,
    2018/8/26,19930,N284AK,686,14057,1405702,34057,PDX,13930,1393006,30977,ORD,700,656.0,-4.0,0.0,1225.0,-34.0,0.0,0,,239,209.0,,,,,
    2018/8/28,20397,N218PS,5116,14092,1409205,34092,PGV,11057,1105703,31057,CLT,533,524.0,-9.0,0.0,624.0,-27.0,0.0,0,,78,60.0,,,,,
    2018/8/30,19393,N765SW,957,10800,1080003,32575,BUR,13796,1379608,32457,OAK,2120,2150.0,30.0,30.0,2254.0,19.0,19.0,0,,75,64.0,0.0,0.0,0.0,0.0,19.0
    2018/8/31,19805,N540UW,2052,13930,1393006,30977,ORD,11057,1105703,31057,CLT,724,718.0,-6.0,0.0,1004.0,-20.0,0.0,0,,120,106.0,,,,,
    2018/8/31,19393,N265WN,207,12896,1289607,32896,LBB,12889,1288903,32211,LAS,1720,1714.0,-6.0,0.0,1721.0,-9.0,0.0,0,,130,127.0,,,,,
    """

    # Question for multi-table association
    # question = "What is the total budget of the projects managed by the R & D Department? Choices:\n A. 1100\nB. 670\nC. 500 \nD. 1650"
    question = "How many airlines land in Buffalo, NY: Buffalo Niagara International?:\n A. 1 \nB. 2\nC. 0 \nD. 3assistant"


    '''
    正确答案
    是A
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
