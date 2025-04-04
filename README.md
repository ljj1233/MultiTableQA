# MultiTableQA

一个基于表格增强的多表问答系统，能够处理复杂的多表关联查询问题。

## 项目结构

- `MTable.py`: 实现表格增强功能的核心逻辑
- `table_qa_evaluator.py`: 主要入口文件，负责模型加载和评估
- `Utils/dataLoader.py`: 数据加载和结果记录工具

## 运行流程

1. **初始化阶段**
   - 加载预训练的 Llama 模型
   - 应用表格增强功能
   - 配置提示类型（原始/思维链/表格增强）

2. **数据加载阶段**
   - 连接任务数据库和结果数据库
   - 加载表格、问题和选项数据

3. **评估循环**
   - 遍历数据集中的每个问题
   - 将表格序列化为文本格式
   - 构建适合的提示模板

4. **推理阶段**
   - 对于表格增强模式，传递表格内容给模型
   - 在生成过程中，当模型遇到高熵状态时触发表格特征注入
   - 表格特征与模型隐藏状态融合，增强表格理解能力

5. **结果记录**
   - 提取模型回答中的最终答案
   - 评估正确性并保存结果

## 使用方法

```bash
python d:\NLP\MultiTableQA\table_qa_evaluator.py \
  --model_path path/to/model \
  --db_root path/to/db \
  --task_path path/to/task.sqlite \
  --result_path path/to/result.sqlite \
  --dataset dataset_name \
  --scale small \
  --prompt_type [default|cot|retrace_table]