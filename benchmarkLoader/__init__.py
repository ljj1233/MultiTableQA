import sys
sys.path.append('.')

# 单个问题的提示模板
singlePrompt = """Please answer the following question:

{question}

Please provide your answer in the format:
Answer: [Your choice letter]
"""

class BenchmarkDataset:
    """
    基础数据集类，所有特定任务的数据集类都应该继承这个类
    """
    def __init__(self, scale=None, markdown=True):
        """
        初始化基础数据集
        
        Args:
            scale: 数据规模，如 '8k', '16k', '32k', '64k', '128k'
            markdown: 是否使用markdown格式输出表格
        """
        self.scale = scale
        self.markdown = markdown
        self.maps = 'A B C D E F'.split()  # 选项映射
        self.taskList = []  # 任务列表，子类需要填充
    
    def loadDB(self, dbName):
        """
        加载数据库，子类可以重写此方法
        
        Args:
            dbName: 数据库名称
            
        Returns:
            数据库对象
        """
        from benchmarkUtils.dbTool import DBLoader
        return DBLoader(dbName)
    
    def __getitem__(self, index):
        """
        获取数据集中的一个样本，子类必须实现此方法
        
        Args:
            index: 样本索引
            
        Returns:
            (问题文本, 正确答案)
        """
        raise NotImplementedError("子类必须实现__getitem__方法")
    
    def __len__(self):
        """
        获取数据集大小，子类必须实现此方法
        
        Returns:
            数据集中样本数量
        """
        return len(self.taskList)