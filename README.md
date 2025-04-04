# MultiTableQA

一个基于表格增强的多表问答系统，能够处理复杂的多表关联查询问题。

## 项目结构

- `multi_Table.py`: 实现表格增强功能的核心逻辑
- `open_model_evaluator.py`: 开源模型评估工具，用于评估本地模型性能
- `api_model_evaluator.py`: API模型评估工具，用于评估OpenAI/Claude等API模型性能
- `eval/`: 评估工具包，包含评估函数和工具
- `Utils/jsTool.py`: 数据加载和结果记录工具

## 运行流程

1. **初始化阶段**
   - 加载预训练的 Llama 模型
   - 应用表格增强功能
   - 配置提示类型（原始/思维链/表格增强）

2. **数据加载阶段**
   - 加载表格、问题和选项数据
   - 准备评估数据集

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

### 1. 评估开源模型

使用`open_model_evaluator.py`评估本地部署的开源模型：

```bash
# 默认提示模式评估
python d:\NLP\MultiTableQA\open_model_evaluator.py \
  --model_path D:\Models\Meta-Llama-3.1-8B-Instruct \
  --dataset qa \
  --scale 16k \
  --prompt_type default

# 思维链(CoT)提示模式评估
python d:\NLP\MultiTableQA\open_model_evaluator.py \
  --model_path D:\Models\Meta-Llama-3.1-8B-Instruct \
  --dataset qa \
  --scale 16k \
  --prompt_type cot \
  --result_path d:\NLP\MultiTableQA\results\qa_cot_result.json

# 表格增强提示模式评估
python d:\NLP\MultiTableQA\open_model_evaluator.py \
  --model_path D:\Models\Meta-Llama-3.1-8B-Instruct \
  --dataset qa \
  --scale 16k \
  --prompt_type retrace_table \
  --markdown \
  --log_root d:\NLP\MultiTableQA\logs\qa_retrace
```