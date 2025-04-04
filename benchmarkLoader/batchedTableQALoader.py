import sys
sys.path.append('.')
from benchmarkLoader import BenchmarkDataset, singlePrompt
from benchmarkUtils.jsTool import JS

class BatchedTableQADataset(BenchmarkDataset):
    """
    批处理表格问答数据集类
    一次处理多个表格问答任务
    """
    def __init__(self, batch_size, scale, markdown=True):
        """
        初始化批处理表格问答数据集
        
        Args:
            batch_size: 批处理大小
            scale: 数据规模，如 '8k', '16k', '32k', '64k', '128k'
            markdown: 是否使用markdown格式输出表格
        """
        super().__init__(scale, markdown)
        jsPath = 'dataset/task/tableQA/task.json'
        self.taskList = JS(jsPath).loadJS()  # 加载任务列表
        self.batch_size = batch_size
        
        # 计算批次数量
        self.num_batches = (len(self.taskList) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        """
        获取一批表格问答样本
        
        Args:
            index: 批次索引
            
        Returns:
            (批次问题文本, 批次正确答案列表)
        """
        # 计算当前批次的起始和结束索引
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.taskList))
        
        batch_questions = []
        batch_answers = []
        
        # 处理批次中的每个样本
        for i in range(start_idx, end_idx):
            qa = self.taskList[i]
            question = qa['question']
            choices = qa['choices']
            
            # 格式化选项
            choiceList = []
            for j in range(len(choices)):
                choiceList.append(f'{self.maps[j]}) {choices[j]}')
            choiceStr = '\n'.join(choiceList)
            
            # 加载并序列化表格
            tables = self.loadDB(qa['database']).defaultSerialization(self.markdown)
            rightChoice = self.maps[qa['rightIdx'][self.scale]]  # 获取正确答案
            
            # 组合单个问题
            single_question = f'# Question {i - start_idx}\n# {qa["database"]}\n\n{tables}\n\n{question}\n\n{choiceStr}'
            batch_questions.append(single_question)
            batch_answers.append(rightChoice)
        
        # 组合批次问题
        combined_question = "\n\n".join(batch_questions)
        combined_question += "\n\nPlease answer all questions in the format:\nAnswer 0: [Your choice letter for question 0]\nAnswer 1: [Your choice letter for question 1]\n..."
        
        return combined_question, batch_answers

    def __len__(self):
        """
        获取批次数量
        
        Returns:
            数据集中批次数量
        """
        return self.num_batches