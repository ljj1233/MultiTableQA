import sys
sys.path.append('.')
from benchmarkLoader import BenchmarkDataset, singlePrompt
from benchmarkUtils.jsTool import JS

class RetrievalDataset(BenchmarkDataset):
    """
    表格检索数据集类
    处理表格检索任务，从多个表格中找出最相关的表格
    """
    def __init__(self, scale, markdown=True):
        """
        初始化表格检索数据集
        
        Args:
            scale: 数据规模，如 '8k', '16k', '32k', '64k', '128k'
            markdown: 是否使用markdown格式输出表格
        """
        super().__init__(scale, markdown)
        jsPath = 'dataset/task/retrieval/task.json'
        self.taskList = JS(jsPath).loadJS()  # 加载任务列表

    def __getitem__(self, index):
        """
        获取一个表格检索样本
        
        Args:
            index: 样本索引
            
        Returns:
            (问题文本, 正确答案)
        """
        ret = self.taskList[index]
        query = ret['query']
        
        # 构建选项和加载表格
        choiceList = []
        tableContents = []
        
        for i, db in enumerate(ret['candidates']):
            # 加载并序列化表格
            table = self.loadDB(db).defaultSerialization(self.markdown)
            tableContents.append(f'# 表格 {self.maps[i]}: {db}\n\n{table}')
            choiceList.append(f'{self.maps[i]}) 表格 {self.maps[i]}')
        
        # 组合所有表格内容
        tablesStr = '\n\n'.join(tableContents)
        choiceStr = '\n'.join(choiceList)
        
        # 获取正确答案
        rightChoice = self.maps[ret['rightIdx'][self.scale]]
        
        # 组合完整问题
        totalQuestion = f'{tablesStr}\n\n请根据以下查询，选择最相关的表格：\n{query}\n\n{choiceStr}'
        totalQuestion = singlePrompt.format(question=totalQuestion)
        
        return totalQuestion, rightChoice