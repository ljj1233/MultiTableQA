import os
import pandas as pd
import sys
sys.path.append('.')
from benchmarkLoader import BenchmarkDataset, singlePrompt
from benchmarkUtils.jsTool import JS

class CPADataset(BenchmarkDataset):
    """
    列属性预测数据集类
    处理列属性预测任务
    """
    def __init__(self, scale, markdown=True):
        """
        初始化列属性预测数据集
        
        Args:
            scale: 数据规模，如 '8k', '16k', '32k', '64k', '128k'
            markdown: 是否使用markdown格式输出表格
        """
        super().__init__(scale, markdown)
        jsPath = f'dataset/task/cpa/{self.scale}/task.json'
        self.taskList = JS(jsPath).loadJS()  # 加载任务列表
        self.tableRoot = 'dataset/sotab-cpa/Validation/'  # 表格根目录

    def __getitem__(self, index):
        """
        获取一个列属性预测样本
        
        Args:
            index: 样本索引
            
        Returns:
            (问题文本, 正确答案)
        """
        item = self.taskList[index]
        columns = item['columns']
        choices = item['choices']
        rightIdx = item['rightIdx']
        
        # 加载表格
        tablePath = os.path.join(self.tableRoot, item['table'])
        df = pd.read_json(tablePath, compression='gzip', lines=True)
        
        # 根据格式要求序列化表格
        dfStr = ''
        if self.markdown:
            dfStr = df.to_markdown(index=False)
        else:
            dfStr = df.to_csv(index=False)

        # 格式化选项
        choiceList = []
        for i in range(4):
            choiceList.append(f'{self.maps[i]}) {choices[i]}')
        choiceStr = '\n'.join(choiceList)
        
        # 组合完整问题
        question = f'{dfStr}\n\nPlease select the relevance from column {columns[0]} to column {columns[1]}.\n\n{choiceStr}'
        question = singlePrompt.format(question=question)
        
        return question, self.maps[rightIdx]