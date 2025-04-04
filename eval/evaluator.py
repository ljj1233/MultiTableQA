import os
import argparse
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import time
from Utils.jsTool import JS
from eval.evaluator_utils import extractAnswer, extractBatchedAnswer, evalFile
from benchmarkLoader.tableQALoader import TableQADataset
from benchmarkLoader.tableFVLoader import TableFVDataset
from benchmarkLoader.retrievalLoader import RetrievalDataset
from benchmarkLoader.cpaLoader import CPADataset
from benchmarkLoader.ctaLoader import CTADataset
from benchmarkLoader.emLoader import EMDataset
from benchmarkLoader.batchedTableQALoader import BatchedTableQADataset

def evalAcc(ds,       # 数据集类型
            scale,    # 数据规模 8k-128k (不适用于em)
            markdown, # 是否使用markdown格式 (不适用于em)
            model,    # 模型名称或路径
            evaluator, # 评估器实例
            prompt_type="default", # 提示类型
            logRoot=None,  # 日志根目录
            resultPath=None, # 结果JSON路径
            is_api_model=False # 是否为API模型
            ):
    """
    评估模型在特定数据集上的准确率
    
    Args:
        ds: 数据集类型
        scale: 数据规模
        markdown: 是否使用markdown格式
        model: 模型名称或路径
        evaluator: 评估器实例
        prompt_type: 提示类型
        logRoot: 日志根目录
        resultPath: 结果JSON路径
        is_api_model: 是否为API模型
    """
    # 导入数据集类型
    
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
        model_name = model if is_api_model else os.path.basename(model.rstrip("/\\"))
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
    
    idx = 0
    saveList = []
    
    # 处理批处理数据集
    if ds.startswith('b'):
        for q, c in tqdm(dataset, desc=ds):
            pred = ['' for _ in range(len(c))]
            err = None
            try:
                # 调用模型
                if is_api_model:
                    res = evaluator.answer_question(
                        q, "", "", None, prompt_type, f'{ds}-{idx}', logRoot
                    )
                else:
                    res = evaluator.answer_question(
                        q, "", "", None, prompt_type
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
                if is_api_model:
                    res = evaluator.answer_question(
                        q, "", "", None, prompt_type, f'{ds}-{idx}', logRoot
                    )
                else:
                    res = evaluator.answer_question(
                        q, "", "", None, prompt_type
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
            
            # API模型添加延迟避免限制
            if is_api_model:
                time.sleep(60)  # 避免API限制

    # 评估结果
    evalFile(resultPath)