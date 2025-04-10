import sqlite3
import pandas as pd
from typing import Dict, List
from collections import defaultdict

class ResultEvaluator:
    def __init__(self, result_path: str):
        """初始化评估器
        Args:
            result_path: 结果数据库路径
        """
        self.conn = sqlite3.connect(result_path)
        self.cursor = self.conn.cursor()
        
    def get_table_names(self) -> List[str]:
        """获取所有表名（即所有dbn）"""
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        return [row[0] for row in self.cursor.fetchall()]

    def evaluate_by_dbidx(self, dbn: str, dbidx: int = None) -> Dict:
        """评估特定数据库的特定dbidx的结果
        
        Args:
            dbn: 数据库名称
            dbidx: 数据库索引，如果为None则评估所有dbidx
        """
        query = f"""
        SELECT 
            dbidx,
            COUNT(*) as total_questions,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_answers,
            scale,
            markdown
        FROM {dbn}
        """
        
        if dbidx is not None:
            query += f" WHERE dbidx = {dbidx}"
            
        query += " GROUP BY dbidx, scale, markdown"
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        metrics = []
        for row in results:
            dbidx, total, correct, scale, markdown = row
            accuracy = (correct / total * 100) if total > 0 else 0
            metrics.append({
                'dbidx': dbidx,
                'total_questions': total,
                'correct_answers': correct,
                'accuracy': accuracy,
                'scale': scale,
                'markdown': markdown
            })
            
        return metrics

    def evaluate_all_dbns(self) -> Dict[str, List[Dict]]:
        """评估所有数据库的结果"""
        all_results = {}
        for dbn in self.get_table_names():
            all_results[dbn] = self.evaluate_by_dbidx(dbn)
        return all_results

    def get_detailed_errors(self, dbn: str, dbidx: int = None) -> pd.DataFrame:
        """获取详细的错误信息
        
        Args:
            dbn: 数据库名称
            dbidx: 可选的数据库索引
        """
        query = f"""
        SELECT 
            dbidx,
            sampleidx,
            questionidx,
            gt as ground_truth,
            pred as prediction,
            error,
            message,
            scale,
            markdown
        FROM {dbn}
        WHERE correct = 0
        """
        
        if dbidx is not None:
            query += f" AND dbidx = {dbidx}"
            
        return pd.read_sql_query(query, self.conn)

def main():
    # 替换为你的结果数据库路径
    result_path = r"c:\Users\Lenovo\Desktop\科研\MLLM\MultiTableQA\results\evaluation_results.sqlite"
    evaluator = ResultEvaluator(result_path)
    
    # 获取所有评估结果
    all_results = evaluator.evaluate_all_dbns()
    
    # 打印评估结果
    for dbn, metrics in all_results.items():
        print(f"\n=== 数据库: {dbn} ===")
        
        # 按scale和markdown分组统计
        stats = defaultdict(lambda: defaultdict(list))
        for m in metrics:
            key = f"{m['scale']}_{'markdown' if m['markdown'] else 'csv'}"
            stats[key]['accuracy'].append(m['accuracy'])
            stats[key]['total'].append(m['total_questions'])
            stats[key]['correct'].append(m['correct_answers'])
        
        # 打印每个配置的统计信息
        for config, values in stats.items():
            avg_accuracy = sum(values['accuracy']) / len(values['accuracy'])
            total_questions = sum(values['total'])
            total_correct = sum(values['correct'])
            
            print(f"\n配置: {config}")
            print(f"平均准确率: {avg_accuracy:.2f}%")
            print(f"总问题数: {total_questions}")
            print(f"正确回答数: {total_correct}")
            
            # 打印每个dbidx的具体信息
            print("\ndbidx详细信息:")
            for i, acc in enumerate(values['accuracy']):
                print(f"dbidx {i}: 准确率 {acc:.2f}%, "
                      f"正确数 {values['correct'][i]}/{values['total'][i]}")
        
        # 获取错误详情
        print("\n错误详情示例:")
        errors = evaluator.get_detailed_errors(dbn)
        if not errors.empty:
            print(errors.head())

if __name__ == "__main__":
    main()