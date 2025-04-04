import sys
import pandas as pd
sys.path.append('.')
from benchmarkLoader import BenchmarkDataset, singlePrompt
from benchmarkUtils.jsTool import JS

class EMDataset(BenchmarkDataset):
    """
    实体匹配数据集类
    处理实体匹配任务，判断两个实体是否指代同一对象
    """
    def __init__(self):
        """
        初始化实体匹配数据集
        注意：EM数据集不需要scale和markdown参数
        """
        super().__init__()
        jsPath = 'dataset/task/em/task.json'
        self.taskList = JS(jsPath).loadJS()  # 加载任务列表

    def __getitem__(self, index):
        """
        获取一个实体匹配样本
        
        Args:
            index: 样本索引
            
        Returns:
            (问题文本, 正确答案)
        """
        em = self.taskList[index]
        entity1 = em['entity1']
        entity2 = em['entity2']
        
        # 将实体转换为DataFrame以便更好地展示
        df1 = pd.DataFrame([entity1])
        df2 = pd.DataFrame([entity2])
        
        df1_str = df1.to_markdown(index=False)
        df2_str = df2.to_markdown(index=False)
        
        # 构建选项
        choiceList = [
            "A) 是同一实体",
            "B) 不是同一实体"
        ]
        choiceStr = '\n'.join(choiceList)
        
        # 获取正确答案
        rightChoice = self.maps[em['rightIdx']]
        
        # 组合完整问题
        totalQuestion = f'# 实体1:\n{df1_str}\n\n# 实体2:\n{df2_str}\n\n请判断以上两个实体是否指代同一对象：\n\n{choiceStr}'
        totalQuestion = singlePrompt.format(question=totalQuestion)
        
        return totalQuestion, rightChoice