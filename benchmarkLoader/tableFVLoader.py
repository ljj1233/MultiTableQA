import sys
sys.path.append('.')
from benchmarkLoader import BenchmarkDataset, singlePrompt
from Utils.jsTool import JS
class TableFVDataset(BenchmarkDataset):
    """
    表格事实验证数据集类
    处理表格事实验证任务，判断陈述是否正确
    """
    def __init__(self, scale, markdown=True):
        """
        初始化表格事实验证数据集
        
        Args:
            scale: 数据规模，如 '8k', '16k', '32k', '64k', '128k'
            markdown: 是否使用markdown格式输出表格
        """
        super().__init__(scale, markdown)
        jsPath = 'dataset/task/tableFV/task.json'
        self.taskList = JS(jsPath).loadJS()  # 加载任务列表

    def __getitem__(self, index):
        """
        获取一个表格事实验证样本
        
        Args:
            index: 样本索引
            
        Returns:
            (问题文本, 正确答案)
        """
        fv = self.taskList[index]
        statement = fv['statement']
        
        # 加载并序列化表格
        tables = self.loadDB(fv['database']).defaultSerialization(self.markdown)
        
        # 构建选项
        choiceList = [
            "A) 正确",
            "B) 错误"
        ]
        choiceStr = '\n'.join(choiceList)
        
        # 获取正确答案
        rightChoice = self.maps[fv['rightIdx'][self.scale]]
        
        # 组合完整问题
        totalQuestion = f'# {fv["database"]}\n\n{tables}\n\n请判断以下陈述是否正确：\n{statement}\n\n{choiceStr}'
        totalQuestion = singlePrompt.format(question=totalQuestion)
        
        return totalQuestion, rightChoice