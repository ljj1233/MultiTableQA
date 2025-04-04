import sys
sys.path.append('.')
from benchmarkLoader import BenchmarkDataset, singlePrompt
from benchmarkUtils.jsTool import JS

class TableQADataset(BenchmarkDataset):
    """
    表格问答数据集类
    处理基于表格的问答任务
    """
    def __init__(self, scale, markdown=True):
        """
        初始化表格问答数据集
        
        Args:
            scale: 数据规模，如 '8k', '16k', '32k', '64k', '128k'
            markdown: 是否使用markdown格式输出表格
        """
        super().__init__(scale, markdown)
        jsPath = 'dataset/task/tableQA/task.json'
        self.taskList = JS(jsPath).loadJS()  # 加载任务列表

    def __getitem__(self, index):
        """
        获取一个表格问答样本
        
        Args:
            index: 样本索引
            
        Returns:
            (问题文本, 正确答案)
        """
        qa = self.taskList[index]
        question = qa['question']
        choices = qa['choices']
        
        # 格式化选项
        choiceList = []
        for i in range(len(choices)):
            choiceList.append(f'{self.maps[i]}) {choices[i]}')
        choiceStr = '\n'.join(choiceList)
        
        # 加载并序列化表格
        tables = self.loadDB(qa['database']).defaultSerialization(self.markdown)
        rightChoice = self.maps[qa['rightIdx'][self.scale]]  # 获取正确答案

        # 组合完整问题
        totalQuestion = f'# {qa["database"]}\n\n{tables}\n\n{question}\n\n{choiceStr}'
        totalQuestion = singlePrompt.format(question=totalQuestion)
        
        return totalQuestion, rightChoice

    def __len__(self):
        """
        获取数据集大小
        
        Returns:
            数据集中样本数量
        """
        return len(self.taskList)