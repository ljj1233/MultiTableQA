import os
import pandas as pd
import sys
sys.path.append('.')
from benchmarkLoader import BenchmarkDataset, singlePrompt
from Utils.jsTool import JS

class CTADataset(BenchmarkDataset):
    """
    列类型分析数据集类
    处理列类型分析任务，判断表格列的数据类型
    """
    def __init__(self, scale, markdown=True):
        """
        初始化列类型分析数据集
        
        Args:
            scale: 数据规模，如 '8k', '16k', '32k', '64k', '128k'
            markdown: 是否使用markdown格式输出表格
        """
        super().__init__(scale, markdown)
        jsPath = f'dataset/task/cta/{self.scale}/task.json'
        self.taskList = JS(jsPath).loadJS()  # 加载任务列表
        self.tableRoot = 'dataset/sotab-cta/Validation/'  # 表格根目录

    def __getitem__(self, index):
        """
        获取一个列类型分析样本
        
        Args:
            index: 样本索引
            
        Returns:
            (问题文本, 正确答案)
        """
        item = self.taskList[index]
        column = item['column']
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
        for i in range(len(choices)):
            choiceList.append(f'{self.maps[i]}) {choices[i]}')
        choiceStr = '\n'.join(choiceList)
        
        # 组合完整问题
        question = f'{dfStr}\n\n请判断列 "{column}" 的数据类型：\n\n{choiceStr}'
        question = singlePrompt.format(question=question)
        
        return question, self.maps[rightIdx]