import re
from Utils.jsTool import JS

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