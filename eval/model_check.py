import re
import os
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import time

from benchmarkUtils.LLM import gptCall
from Utils.jsTool import JS
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
            model,    # 模型类型 gpt-4, gpt-4o, gpt-4o-mini
            logRoot,  # 日志根目录
            resultPath # 结果JSON路径
            ):
    """
    评估模型在特定数据集上的准确率
    
    Args:
        ds: 数据集类型
        scale: 数据规模
        markdown: 是否使用markdown格式
        model: 模型类型
        logRoot: 日志根目录
        resultPath: 结果JSON路径
    """
    global dsDict
    if ds not in dsDict.keys():
        return None
        
    # 设置默认日志目录
    if logRoot == None:
        logRoot = os.path.join('results', ds)
        
    # 设置默认结果路径
    if resultPath == None:
        tmp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + "_" + str(uuid4()) + ".json"
        resultName = f'{ds}_{scale}_{markdown}_{model}_{tmp}'
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
                res = gptCall(
                    model,
                    q,
                    f'{ds}-{idx}',
                    logRoot
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
                res = gptCall(
                    model,
                    q,
                    f'{ds}-{idx}',
                    logRoot
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
            time.sleep(60)  # 避免API限制

    # 评估结果
    evalFile(resultPath)


if __name__ == '__main__':
    evalAcc(
        'qa',      # 数据集类型
        '16k',     # 数据规模
        True,      # 使用markdown格式
        'gpt-4o',  # 模型类型
        None,      # 使用默认日志目录
        None       # 使用默认结果路径
    )
